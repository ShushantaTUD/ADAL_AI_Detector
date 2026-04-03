"""
RADAR × RAID — Multi-Generator + Multi-Evasion Adversarial Training
====================================================================
Paper : "RADAR: Robust AI-Text Detection via Adversarial Learning"
        Hu et al., NeurIPS 2023 — https://arxiv.org/pdf/2307.03838
Dataset: RAID (liamdugan/raid) — Dugan et al., ACL 2024

═══════════════════════════════════════════════════════════════════
DESIGN PHILOSOPHY: TWO-TRACK ADVERSARIAL GAME
═══════════════════════════════════════════════════════════════════

Original RADAR has ONE evasion strategy: the learnable T5 paraphraser (Gσ).
We extend it with an EvasionAttackPool of 6 strategies, split into two tracks:

  Track A — LEARNABLE (PPO)
  ─────────────────────────
  T5-Paraphrase      : Gσ rewrites AI text via seq2seq. Updated via cppo-ep.
                       This is the only attack for which log-probs exist, so
                       it is the only one that feeds the PPO gradient.

  Track B — DETERMINISTIC (Data Augmentation for Detector)
  ──────────────────────────────────────────────────────────
  RecursiveParaphrase: T5 applied N times in a chain (xm→xp1→xp2…).
                       Increases linguistic distance from original AI text.
  SynonymReplacement : Replaces content words with WordNet synonyms at rate p.
                       Mimics manual word-swap evasion (common in student essays).
  Homoglyphs         : Swaps ASCII chars with visually identical Unicode chars
                       (e.g. 'a'→'а' Cyrillic). Exploits tokeniser blind spots.
  ArticleDeletion    : Removes 'a','an','the' randomly at rate p. Non-native
                       English writers do this naturally; detectors misfire.
  RandomMisspelling  : Inserts adjacent-key typos at rate p per word.
                       Simulates rushed human writing and confuses n-gram signals.

Training loop per outer step:
  1. Sample xm from generator pool.
  2. Track A: Gσ generates xp_ppo  → PPO reward → update Gσ.
  3. Track B: ALL 5 deterministic attacks produce xp_det_i for each xm.
  4. Detector update: sees xh vs {xm, xp_ppo, xp_det_1…xp_det_5}.
     This is a multi-attack reweighted logistic loss (Eq. 3 extended).

Why this matters
────────────────
  • Gσ is trained against the current Dϕ → learns neural evasion.
  • Dϕ is trained against ALL attacks simultaneously → robust to the full
    threat model, not just T5 paraphrasing.
  • At test time, ONLY Dϕ is deployed.  It has seen every attack during
    training, generalising far beyond the original RADAR paper.

Stability fixes from previous iteration (all retained)
────────────────────────────────────────────────────────
  Fix 1 — Label smoothing (LABEL_SMOOTHING_ALPHA=0.10)
  Fix 2 — Detector update frequency (DETECTOR_UPDATE_EVERY=2)
  Fix 3 — Reward clipping [REWARD_CLIP_MIN, REWARD_CLIP_MAX]
  Fix 4 — Paraphraser MLE warm-start (WARMSTART_STEPS=5)
  Fix 5 — KL penalty on PPO (KL_COEFF=0.1)

Dependencies
────────────
  pip install transformers datasets scikit-learn torch sentencepiece nltk
  python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
"""

import os
import math

# Fix OOM-5: reduce fragmentation from mixed T5/RoBERTa allocation pattern
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re
import random
import string
import logging
import unicodedata
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from transformers import LogitsProcessor, LogitsProcessorList
from raid.utils import load_data as raid_load_data   # official RAID train/test splits


class NanSafeLogitsProcessor(LogitsProcessor):
    """Intercept logits before torch.multinomial — replace NaN/inf so sampling never crashes."""
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e4, neginf=-1e9)
        all_dead = (scores <= -1e8).all(dim=-1)
        if all_dead.any():
            scores[all_dead] = 0.0   # uniform fallback — all tokens equally likely
        return scores

_NAN_SAFE_PROCESSOR = LogitsProcessorList([NanSafeLogitsProcessor()])
from sklearn.metrics import roc_auc_score
from huggingface_hub import HfApi, login

# NLTK imports (graceful fallback if not installed)
try:
    from nltk.corpus import wordnet
    from nltk import pos_tag, word_tokenize
    import nltk
    nltk.download("wordnet",                      quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt",                         quiet=True)
    nltk.download("punkt_tab",                     quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION — no argparse, edit these variables only
# ══════════════════════════════════════════════════════════════════════════════

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model identifiers ─────────────────────────────────────────────────────────
# Fine-tuned paraphrase model — much better evasion diversity than vanilla t5-large.
# "Vamsi/T5_Paraphrase_Paws" is trained on PAWS paraphrase pairs; understands
# "Paraphrase: <text>" prefix natively. Swap back to "t5-large" if download fails.
# Vamsi/T5_Paraphrase_Paws generates over-formal text that scores LOWER (more AI-like)
# than the original. Switch to DIPPER (T5-XXL, ~22GB fp16) — the dedicated diversity
# paraphraser from Krishna et al. 2023, the standard evasion attack in the RAID paper.
# Fallback: "humarin/chatgpt_paraphraser_on_T5_base" if VRAM is tight.
# T5-base fine-tuned for paraphrasing — ~900 MB, clean checkpoint, no tied-weight
# inconsistencies. humarin/chatgpt_paraphraser_on_T5_base has mismatched tied weights
# (lm_head/encoder/decoder embed tokens all present but untied) causing gradient
# explosion in warmstart (MLE→NaN on step 2) which then poisons generate() logits.
PARAPHRASER_MODEL_NAME   = "ramsrigouthamg/t5_paraphraser"
PARAPHRASER_LOAD_IN_8BIT = False  # not needed — model is only ~900 MB
DETECTOR_MODEL_NAME    = "roberta-large"

# ── RAID dataset ──────────────────────────────────────────────────────────────
# Install:  pip install raid-bench
# train split → adversarial training  |  test split → validation (no leakage)
GENERATORS_TO_USE: Optional[List[str]] = None   # None = all generators found in data
RAID_FILTER_DOMAIN: Optional[str]      = None   # None = all domains
RAID_FILTER_ATTACK: Optional[str]      = "none" # TRAIN only: use clean (attack="none") rows.
                                                 # The TEST split contains ONLY adversarial rows
                                                 # (no attack="none"), so this filter is skipped
                                                 # for validation — see load_raid_multigenerator().

# ── Sample sizes ──────────────────────────────────────────────────────────────
# Caps applied to the TRAIN split before the train/val split is performed.
# Set None to use all available data.
NUM_HUMAN_SAMPLES     = 10000  # 10,000 human texts total (split into train+val below)
SAMPLES_PER_GENERATOR = 1000   # 1,000 AI texts per generator (split into train+val below)

# Train/val split ratio — applied AFTER sampling the caps above.
# e.g. 0.9 → 90% train, 10% val of the NUM_HUMAN_SAMPLES / SAMPLES_PER_GENERATOR totals.
# The RAID official test split is NOT used for validation: it has NO human texts and uses
# a different schema (empty model field), making it unsuitable as a validation set.
VAL_SPLIT_RATIO = 0.1  # → ~9,000 train / ~1,000 val human; ~900 train / ~100 val per gen

# ── Multi-generator sampling strategy ────────────────────────────────────────
# "uniform" | "round_robin" | "mixed" | "curriculum"
GENERATOR_SAMPLING_STRATEGY = "mixed"
CURRICULUM_SWITCH_STEPS      = 20

# ══════════════════════════════════════════════════════════════════════════════
#  EVASION ATTACK POOL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Which attacks are active.  Setting to False disables the attack entirely.
ATTACK_T5_PARAPHRASE      = True   # Track A: learnable (PPO)
ATTACK_RECURSIVE_PARA     = False  # SPEED: CPU T5 was ~15-18 min/step; disabled.
                                   # Re-enable if you have time budget or run T5 on GPU.
ATTACK_SYNONYM_REPLACEMENT = True
ATTACK_HOMOGLYPHS          = True
ATTACK_ARTICLE_DELETION    = True
ATTACK_MISSPELLING         = True

# Recursive paraphrase: number of T5 passes (1 = same as single paraphrase)
RECURSIVE_PARA_DEPTH = 2

# Synonym replacement rate (fraction of content words replaced per sentence)
SYNONYM_REPLACE_RATE = 0.20

# Homoglyph substitution rate (fraction of eligible chars replaced)
HOMOGLYPH_RATE = 0.10

# Article deletion rate (fraction of articles removed)
ARTICLE_DELETE_RATE = 0.50

# Random misspelling rate (fraction of words that get a typo inserted)
MISSPELLING_RATE = 0.08

# Detector loss weight per attack type (relative to original AI-text weight λ)
# Sum should roughly equal 1.0 for balanced multi-attack training.
# Higher weight = detector sees more examples of that attack type.
ATTACK_LOSS_WEIGHTS: Dict[str, float] = {
    "t5_paraphrase":       1.0,
    "recursive_para":      0.8,
    "synonym_replacement": 0.7,
    "homoglyphs":          0.5,
    "article_deletion":    0.5,
    "misspelling":         0.5,
}

# ── PPO / Paraphraser ─────────────────────────────────────────────────────────
PPO_BUFFER_SIZE            = 64    # SPEED: was 128; halved → ~2× faster buffer fill + PPO
# 8 epochs per buffer — more gradient steps per collection cycle,
# helping the paraphraser learn faster against a strong detector.
PPO_EPOCHS                 = 8
PPO_EPSILON                = 0.2
ENTROPY_COEFF              = 0.01  # FIX: was 0.05; para loss = -136 at step 2 means
                                   # entropy term was dominating. 0.01 matches original paper.
PARAPHRASER_LR             = 2e-5   # BALANCE: 2x faster than before; essential to outpace detector
PARAPHRASER_MAX_NEW_TOKENS = 128
# temperature=1.5 caused NaN probs → multinomial crash. temperature=1.0 is stable.
# top_k=50 with top_p disabled keeps a focused but varied distribution.
# Beam search (do_sample=False) killed PPO: zero reward variance → para_loss=0 always.
PARAPHRASER_TOP_K          = 50     # restored — focused distribution, stable sampling
PARAPHRASER_TOP_P          = 1.0    # keep disabled
PARAPHRASER_TEMPERATURE    = 1.0    # safe — temperature=1.5 caused NaN/crash
PARAPHRASER_REPETITION_PEN = 1.3    # penalise repeating the same tokens
PARAPHRASER_MAX_INPUT_LEN  = 256
# FIX: was 0.1 — too strong. Keeps paraphraser close to reference T5, limiting
# the diversity of generated paraphrases and collapsing reward variance.
# 0.01 allows more deviation so PPO has meaningful gradient signal.
# Lower KL allows paraphraser to deviate more from reference → more reward variance
KL_COEFF                   = 0.001

# ── Detector ──────────────────────────────────────────────────────────────────
LAMBDA                = 0.5
# 1e-5 caused detector to memorize val set in 4 steps (AUROC 0.80→0.99).
# 3e-6 gives slower, more stable learning that stays competitive with paraphraser.
DETECTOR_LR           = 3e-6
DETECTOR_BATCH        = 8     # Fix OOM-1: halved to cut RoBERTa activation memory
ATTACK_MICRO_BATCH    = 4     # Fix OOM-2: micro-batch for per-attack loss groups
DETECTOR_MAX_LEN      = 512
# FIX: was 0.30 — too high. All loss components stuck at ~0.62 = near-random.
# 0.30 smoothing caps detector confidence at 0.70 and kills gradient magnitude.
# 0.10 gives much stronger signal while still preventing collapse.
# Higher smoothing = stronger regularization against memorization.
# Detector jumped 0.80→0.99 in 4 steps — needs more regularization.
LABEL_SMOOTHING_ALPHA = 0.15
# Update detector every 2 steps — updating every step caused memorization
# of the 9900-sample training set in just 4 steps (AUROC 0.80→0.99).
DETECTOR_UPDATE_EVERY = 2
# Dynamic freeze: if macro AUROC exceeds this threshold the detector update
# is skipped for the next DETECTOR_FREEZE_STEPS steps, giving the paraphraser
# room to catch up before the detector is allowed to train again.
# Threshold 0.97 still fired every step (AUROC hit 0.9913 at step 5).
# 0.995 only freezes when detector is near-perfect — harder to trigger.
AUROC_FREEZE_THRESHOLD  = 0.995
DETECTOR_FREEZE_STEPS   = 3

# ── Reward clipping ───────────────────────────────────────────────────────────
# Wider clip range — rewards were stuck at 0.31 which is well within [0.05,0.95]
# so clipping was not the issue. Keep wide to not interfere with natural signal.
REWARD_CLIP_MIN = 0.01
REWARD_CLIP_MAX = 0.99

# ── Warm-start ────────────────────────────────────────────────────────────────
# Warmstart is disabled: some T5-base checkpoints have gradient instability on step 1
# (MLE→NaN) which poisons all subsequent generate() calls with NaN logits.
# PPO works fine without warmstart — the paraphraser learns from reward signal directly.
WARMSTART_STEPS = 0

# ── Training loop ─────────────────────────────────────────────────────────────
MAX_OUTER_STEPS = 200
WARMUP_STEPS    = 50
VALIDATE_EVERY  = 5    # BALANCE: validate more often so we can catch the peak before collapse
PATIENCE        = 25   # BALANCE: more patience since reward oscillates in a healthy game
GRAD_CLIP       = 1.0

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR            = "./radar_multievasion"
DETECTOR_SAVE_PATH    = os.path.join(OUTPUT_DIR, "best_detector")
PARAPHRASER_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_paraphraser")
LOG_FILE              = os.path.join(OUTPUT_DIR, "training.log")
AUROC_LOG_FILE        = os.path.join(OUTPUT_DIR, "per_generator_auroc.tsv")
ATTACK_AUROC_LOG      = os.path.join(OUTPUT_DIR, "per_attack_auroc.tsv")

# ── HuggingFace Hub — push trained models here ────────────────────────────────
# Get your token from: https://huggingface.co/settings/tokens (write access needed)
HF_TOKEN              = "hf_YOUR_TOKEN_HERE"        # ← paste your HF write token
HF_USERNAME           = "your-hf-username"   # ← MUST match exactly: huggingface.co/settings/profile
                                              #   The 403 error means this is wrong or the org
                                              #   namespace does not exist. Use your personal handle.

# Repo IDs: will be created automatically if they don't exist
# Final repo URLs:
#   https://huggingface.co/{HF_USERNAME}/{HF_DETECTOR_REPO}
#   https://huggingface.co/{HF_USERNAME}/{HF_PARAPHRASER_REPO}
HF_DETECTOR_REPO      = "radar-roberta-detector"     # ← repo name for the detector
HF_PARAPHRASER_REPO   = "radar-t5-paraphraser"       # ← repo name for the paraphraser

# When to push:
#   "best"   → push only when a new best AUROC checkpoint is saved (recommended)
#   "final"  → push only at the end of training
#   "both"   → push on every new best AND at end of training
HF_PUSH_STRATEGY      = "both"

# Set False to skip HF push entirely (e.g. for quick debug runs)
HF_PUSH_ENABLED       = True

# ══════════════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
#  EVASION ATTACK POOL
# ══════════════════════════════════════════════════════════════════════════════

# ─── Homoglyph mapping ────────────────────────────────────────────────────────
# ASCII char → list of visually near-identical Unicode alternatives.
# Covers Cyrillic, Greek, and other scripts that look identical in many fonts.
HOMOGLYPH_MAP: Dict[str, List[str]] = {
    'a': ['а', 'ɑ', 'α'],   # Cyrillic а, Latin alpha, Greek alpha
    'c': ['с', 'ϲ'],         # Cyrillic с, Greek lunate sigma
    'e': ['е', 'ė'],         # Cyrillic е
    'i': ['і', 'і'],         # Cyrillic і
    'j': ['ϳ'],              # Cyrillic ϳ
    'o': ['о', 'ο', '０'],   # Cyrillic о, Greek omicron, fullwidth 0
    'p': ['р', 'ρ'],         # Cyrillic р, Greek rho
    's': ['ѕ'],              # Cyrillic ѕ
    'u': ['υ'],              # Greek upsilon
    'x': ['х', 'χ'],         # Cyrillic х, Greek chi
    'y': ['у', 'ý'],         # Cyrillic у
    'B': ['В', 'Β'],         # Cyrillic В, Greek Beta
    'C': ['С', 'Ϲ'],
    'E': ['Е', 'Ε'],         # Cyrillic Е, Greek Epsilon
    'H': ['Н', 'Η'],         # Cyrillic Н, Greek Eta
    'I': ['І', 'Ι'],
    'K': ['К', 'Κ'],         # Cyrillic К, Greek Kappa
    'M': ['М', 'Μ'],
    'N': ['Ν'],              # Greek Nu
    'O': ['О', 'Ο', '0'],   # Cyrillic О, Greek Omicron
    'P': ['Р', 'Ρ'],         # Cyrillic Р, Greek Rho
    'T': ['Т', 'Τ'],
    'X': ['Х', 'Χ'],
    'Z': ['Ζ'],              # Greek Zeta
}

# Adjacent keys on a QWERTY keyboard for realistic typo generation
QWERTY_ADJACENT: Dict[str, str] = {
    'q':'wa',  'w':'qeas', 'e':'wsdr', 'r':'edft', 't':'rfgy', 'y':'tghu',
    'u':'yhji', 'i':'ujko', 'o':'iklp', 'p':'ol',
    'a':'qwsz', 's':'awedxz', 'd':'serfcx', 'f':'drtgvc', 'g':'ftyhbv',
    'h':'gyujnb', 'j':'huikmn', 'k':'jiolm', 'l':'kop',
    'z':'asx',  'x':'zsdc', 'c':'xdfv', 'v':'cfgb', 'b':'vghn',
    'n':'bhjm', 'm':'njk',
}

ARTICLES = {'a', 'an', 'the'}


class AttackResult(NamedTuple):
    text:        str
    attack_name: str


class EvasionAttackPool:
    """
    Pool of 6 text evasion attacks (1 learnable, 5 deterministic).
    All deterministic attacks are pure Python — no GPU required.
    They run in milliseconds and are applied in-process during buffer filling.

    Attack taxonomy
    ───────────────
    Neural (Track A, PPO-trained):
      t5_paraphrase      — Gσ rewrites the text via seq2seq

    Linguistic (Track B, data augmentation for detector):
      recursive_para     — chain T5 paraphrase N times
      synonym_replacement— WordNet-based word swap
      article_deletion   — drop 'a/an/the' tokens
      misspelling        — QWERTY adjacent-key typo injection

    Character-level (Track B):
      homoglyphs         — swap ASCII → visually identical Unicode
    """

    def __init__(self, paraphraser_model=None, paraphraser_tokenizer=None):
        """
        paraphraser_model / tokenizer: the T5 model instance.
        Pass None to disable neural attacks (pure deterministic mode).
        """
        self.t5_model     = paraphraser_model
        self.t5_tokenizer = paraphraser_tokenizer

    # ─── Track A: Neural T5 paraphrase ───────────────────────────────────────
    @torch.no_grad()
    def t5_paraphrase(self, texts: List[str], depth: int = 1) -> List[str]:
        """
        Apply T5 paraphrase `depth` times in a chain.
        depth=1 → standard single-pass paraphrase (same as original RADAR).
        depth=2 → recursive double paraphrase.
        """
        if self.t5_model is None:
            return texts   # fallback: no-op

        current = texts
        for _ in range(depth):
            # Use the model's ACTUAL device — may be CPU when offloaded
            model_device = next(self.t5_model.parameters()).device
            # Use model-aware prefix — ramsrigouthamg uses lowercase "paraphrase: "
            prefix   = "paraphrase: " if any(x in PARAPHRASER_MODEL_NAME
                           for x in ("humarin", "ramsrigouthamg")) else "Paraphrase: "
            prefixed = [f"{prefix}{t}" for t in current]
            enc = self.t5_tokenizer(
                prefixed, return_tensors="pt", padding=True,
                truncation=True, max_length=PARAPHRASER_MAX_INPUT_LEN,
            ).to(model_device)
            first_param = next(self.t5_model.parameters())
            if torch.isnan(first_param).any():
                raise RuntimeError("Attack pool T5 weights contain NaN — model corrupted.")
            # Use beam search (do_sample=False) for the attack pool — this is only
            # used for AUROC measurement, not PPO gradient computation.
            # Beam search never calls torch.multinomial so it CANNOT crash with the
            # "inf/nan in probability tensor" error that stochastic sampling can hit.
            try:
                gen_ids = self.t5_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=PARAPHRASER_MAX_NEW_TOKENS,
                    do_sample=False,          # beam search — no multinomial, no NaN crash
                    num_beams=4,
                    repetition_penalty=PARAPHRASER_REPETITION_PEN,
                    pad_token_id=self.t5_tokenizer.pad_token_id,
                    early_stopping=True,
                )
                current = [
                    self.t5_tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in gen_ids
                ]
            except RuntimeError as e:
                logger.warning(f"  attack_pool.t5_paraphrase generate failed: {e} — returning originals")
                current = current  # return input unchanged if generate crashes
        return current

    # ─── Synonym Replacement ─────────────────────────────────────────────────
    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> Optional[str]:
        """Convert Penn Treebank POS tag to WordNet POS."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        if treebank_tag.startswith('V'):
            return wordnet.VERB
        if treebank_tag.startswith('N'):
            return wordnet.NOUN
        if treebank_tag.startswith('R'):
            return wordnet.ADV
        return None

    @staticmethod
    def synonym_replace(text: str, rate: float = SYNONYM_REPLACE_RATE) -> str:
        """
        Replace content words (N/V/ADJ/ADV) with a random WordNet synonym.
        Preserves original capitalisation and non-content tokens.

        Example:
            "The quick brown fox"  →  "The swift brown fox"
        """
        if not NLTK_AVAILABLE:
            return text
        try:
            tokens   = word_tokenize(text)
            pos_tags = pos_tag(tokens)
        except Exception:
            return text

        new_tokens = []
        for word, tag in pos_tags:
            wn_pos = EvasionAttackPool._get_wordnet_pos(tag)
            if wn_pos and random.random() < rate:
                synsets = wordnet.synsets(word, pos=wn_pos)
                synonyms = [
                    lemma.name().replace('_', ' ')
                    for syn in synsets
                    for lemma in syn.lemmas()
                    if lemma.name().lower() != word.lower()
                       and '_' not in lemma.name()   # exclude multi-word
                ]
                if synonyms:
                    chosen = random.choice(synonyms)
                    # Preserve capitalisation
                    if word[0].isupper():
                        chosen = chosen.capitalize()
                    new_tokens.append(chosen)
                    continue
            new_tokens.append(word)
        return ' '.join(new_tokens)

    # ─── Homoglyphs ───────────────────────────────────────────────────────────
    @staticmethod
    def homoglyph_replace(text: str, rate: float = HOMOGLYPH_RATE) -> str:
        """
        Randomly replace ASCII characters with visually identical Unicode
        homoglyphs. The text looks identical to humans but changes the token
        sequence seen by the detector's tokeniser.

        Only substitutes characters that have a known safe homoglyph.
        Replaces at most `rate` fraction of eligible characters.

        Example (ASCII 'o' → Cyrillic 'о'):
            "Hello world" → "Hellо wоrld"  (two characters changed)
        """
        chars = list(text)
        eligible = [
            i for i, ch in enumerate(chars) if ch in HOMOGLYPH_MAP
        ]
        n_replace = max(1, int(len(eligible) * rate))
        targets   = random.sample(eligible, min(n_replace, len(eligible)))
        for idx in targets:
            ch = chars[idx]
            chars[idx] = random.choice(HOMOGLYPH_MAP[ch])
        return ''.join(chars)

    # ─── Article Deletion ─────────────────────────────────────────────────────
    @staticmethod
    def article_deletion(text: str, rate: float = ARTICLE_DELETE_RATE) -> str:
        """
        Delete articles ('a', 'an', 'the') with probability `rate`.
        This mimics the writing style of non-native English speakers, who
        frequently omit articles.  The RADAR paper (§1) notes that detectors
        show bias against non-native writers; this attack exploits that.

        Example:
            "The cat sat on a mat" → "cat sat on mat"  (rate=1.0)
        """
        tokens    = text.split()
        new_tokens = []
        for tok in tokens:
            if tok.lower() in ARTICLES and random.random() < rate:
                continue   # drop the article
            new_tokens.append(tok)
        return ' '.join(new_tokens)

    # ─── Random Misspelling ───────────────────────────────────────────────────
    @staticmethod
    def random_misspelling(text: str, rate: float = MISSPELLING_RATE) -> str:
        """
        Introduce realistic typos via QWERTY adjacent-key substitution.
        Only affects words longer than 3 chars.  One typo per affected word.

        Types of insertion (randomly chosen for each typo):
          • adjacent-key swap   : 'e' → 'w' or 'r'  (most common real typo)
          • character deletion  : removes a random interior char
          • character duplication: doubles a random interior char

        Example:
            "detection" → "dwtection"  (e→w swap)
        """
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 3 and random.random() < rate:
                typo_type = random.choice(['swap', 'delete', 'duplicate'])
                chars = list(word)
                idx   = random.randint(1, len(chars) - 2)   # avoid first/last

                if typo_type == 'swap':
                    ch = chars[idx].lower()
                    neighbors = QWERTY_ADJACENT.get(ch, '')
                    if neighbors:
                        replacement = random.choice(neighbors)
                        if chars[idx].isupper():
                            replacement = replacement.upper()
                        chars[idx] = replacement

                elif typo_type == 'delete':
                    chars.pop(idx)

                elif typo_type == 'duplicate':
                    chars.insert(idx, chars[idx])

                new_words.append(''.join(chars))
            else:
                new_words.append(word)
        return ' '.join(new_words)

    # ─── Apply all deterministic attacks to a batch ───────────────────────────
    def apply_all_deterministic(
        self,
        ai_texts: List[str],
    ) -> Dict[str, List[str]]:
        """
        Apply every enabled deterministic attack to the batch of AI texts.
        Returns a dict mapping attack_name → list of evaded texts.

        These outputs are used ONLY for detector training (Track B).
        They do not participate in PPO updates.
        """
        results: Dict[str, List[str]] = {}

        if ATTACK_RECURSIVE_PARA and self.t5_model is not None:
            results["recursive_para"] = self.t5_paraphrase(
                ai_texts, depth=RECURSIVE_PARA_DEPTH
            )

        if ATTACK_SYNONYM_REPLACEMENT and NLTK_AVAILABLE:
            results["synonym_replacement"] = [
                self.synonym_replace(t) for t in ai_texts
            ]

        if ATTACK_HOMOGLYPHS:
            results["homoglyphs"] = [
                self.homoglyph_replace(t) for t in ai_texts
            ]

        if ATTACK_ARTICLE_DELETION:
            results["article_deletion"] = [
                self.article_deletion(t) for t in ai_texts
            ]

        if ATTACK_MISSPELLING:
            results["misspelling"] = [
                self.random_misspelling(t) for t in ai_texts
            ]

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  RAID Data Loader
# ══════════════════════════════════════════════════════════════════════════════
def _parse_raid_df(
    df,
    split_name: str,
    n_human: Optional[int],
    n_ai_per_gen: Optional[int],
    attack_filter: Optional[str] = None,    # None = accept all attacks
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Convert a RAID pandas DataFrame (from raid.utils.load_data) into the
    flat lists used by the training loop.

    Filters applied:
      - RAID_FILTER_DOMAIN : keep only this domain if set
      - attack_filter      : if set, keep only rows with this attack value.
                             Pass RAID_FILTER_ATTACK (="none") for training.
                             Pass None for the test split — the RAID test split
                             contains ONLY adversarially attacked rows (no "none"),
                             so filtering by "none" yields 0 rows.
      - GENERATORS_TO_USE  : whitelist of generator names (None = all)
      - Minimum 20 words per text
      - n_human / n_ai_per_gen caps (None = unlimited)
    """
    logger.info(f"  Parsing RAID {split_name} split  "
                f"({len(df):,} rows before filters) …")
    logger.info(f"  Filters → domain={RAID_FILTER_DOMAIN!r}  "
                f"attack={attack_filter!r}  generators={GENERATORS_TO_USE!r}")

    human_texts: List[str]            = []
    ai_pool:     Dict[str, List[str]] = defaultdict(list)

    # For the test split we collect one text per (model, attack) pair per document
    # to keep per-attack AUROC balanced — shuffle handles randomisation.
    for _, row in df.iterrows():
        text   = str(row.get("generation") or "").strip()
        model  = str(row.get("model",  ""))
        domain = str(row.get("domain", ""))
        attack = str(row.get("attack", ""))

        if len(text.split()) < 20:
            continue
        if RAID_FILTER_DOMAIN and domain != RAID_FILTER_DOMAIN:
            continue
        if attack_filter and attack != attack_filter:
            continue

        if model == "human":
            if n_human is None or len(human_texts) < n_human:
                human_texts.append(text)
        else:
            if GENERATORS_TO_USE and model not in GENERATORS_TO_USE:
                continue
            if n_ai_per_gen is None or len(ai_pool[model]) < n_ai_per_gen:
                ai_pool[model].append(text)

    random.shuffle(human_texts)
    for m in ai_pool:
        random.shuffle(ai_pool[m])

    logger.info(f"  Human texts           : {len(human_texts)}")
    logger.info(f"  AI generators found   : {sorted(ai_pool.keys())}")
    for m, v in sorted(ai_pool.items()):
        logger.info(f"    {m:20s} → {len(v)} texts")
    logger.info(f"  Total AI texts        : {sum(len(v) for v in ai_pool.values())}")

    return human_texts, dict(ai_pool)


def _split_list(lst: List, val_ratio: float, seed: int = 42):
    """Shuffle and split a list into (train_part, val_part)."""
    rng = random.Random(seed)
    lst = lst[:]
    rng.shuffle(lst)
    n_val = max(1, int(len(lst) * val_ratio))
    return lst[n_val:], lst[:n_val]


def load_raid_multigenerator() -> Tuple[
    List[str], List[str],
    Dict[str, List[str]], Dict[str, List[str]],
]:
    """
    Load the RAID train split only and carve out a validation set internally.

    Why NOT the official RAID test split:
      - It contains NO human texts (only adversarially attacked AI rows).
      - The "model" column is an empty string ("") instead of generator names,
        so generators cannot be matched to training generators.

    Strategy:
      1. Load RAID train split (attack="none" only — clean AI + human texts).
      2. Apply NUM_HUMAN_SAMPLES / SAMPLES_PER_GENERATOR caps.
      3. Split each list (human + per-generator AI) 90/10 by VAL_SPLIT_RATIO
         using a fixed random seed so every restart uses the SAME val set.

    Install dependency:  pip install raid-bench
    """
    logger.info("Loading RAID dataset via raid.utils.load_data …")
    logger.info("  (install with: pip install raid-bench)")
    logger.info(f"  Caps: human={NUM_HUMAN_SAMPLES}  AI={SAMPLES_PER_GENERATOR}/gen  "
                f"val_ratio={VAL_SPLIT_RATIO}")

    # Load only the train split with clean (non-adversarial) rows
    train_df = raid_load_data(split="train", include_adversarial=False)
    logger.info(f"  RAID train rows (attack='none'): {len(train_df):,}")

    # Parse with caps — _parse_raid_df shuffles internally
    human_all, ai_all_by_model = _parse_raid_df(
        train_df, "full_train", NUM_HUMAN_SAMPLES, SAMPLES_PER_GENERATOR,
        attack_filter=RAID_FILTER_ATTACK,
    )

    # Split humans
    human_train, human_val = _split_list(human_all, VAL_SPLIT_RATIO)

    # Split each generator's AI texts independently
    ai_train_by_model: Dict[str, List[str]] = {}
    ai_val_by_model:   Dict[str, List[str]] = {}
    for gen, texts in ai_all_by_model.items():
        ai_train_by_model[gen], ai_val_by_model[gen] = _split_list(texts, VAL_SPLIT_RATIO)

    logger.info(
        f"  Train → human={len(human_train)}  "
        f"AI={sum(len(v) for v in ai_train_by_model.values())}"
    )
    logger.info(
        f"  Val   → human={len(human_val)}  "
        f"AI={sum(len(v) for v in ai_val_by_model.values())}"
    )
    return human_train, human_val, ai_train_by_model, ai_val_by_model


# ══════════════════════════════════════════════════════════════════════════════
#  Generator Scheduler
# ══════════════════════════════════════════════════════════════════════════════
class GeneratorScheduler:
    def __init__(self, generators: List[str], strategy: str):
        self.generators          = generators
        self.strategy            = strategy
        self._rr_iter            = itertools.cycle(generators)
        self._curr_idx           = 0
        self._steps_on_current   = 0
        self._curriculum_order   = list(generators)
        logger.info(f"GeneratorScheduler: strategy='{strategy}'  generators={generators}")

    def next(self, outer_step: int) -> List[str]:
        if self.strategy == "uniform":
            return [random.choice(self.generators)]
        if self.strategy == "round_robin":
            return [next(self._rr_iter)]
        if self.strategy == "mixed":
            return list(self.generators)
        if self.strategy == "curriculum":
            self._steps_on_current += 1
            if self._steps_on_current > CURRICULUM_SWITCH_STEPS:
                self._steps_on_current = 0
                self._curr_idx = (self._curr_idx + 1) % len(self._curriculum_order)
                logger.info(
                    f"  [Curriculum] → '{self._curriculum_order[self._curr_idx]}'"
                )
            return [self._curriculum_order[self._curr_idx]]
        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def update_curriculum_order(self, auroc_by_generator: Dict[str, float]) -> None:
        ranked = sorted(auroc_by_generator.items(), key=lambda kv: kv[1])
        self._curriculum_order = [g for g, _ in ranked]
        logger.info(
            "  [Curriculum] easy→hard: "
            + "  ".join(f"{g}={a:.3f}" for g, a in ranked)
        )


def sample_from_generators(
    ai_train_by_model: Dict[str, List[str]],
    active_generators:  List[str],
    n: int,
) -> Tuple[List[str], List[str]]:
    if len(active_generators) == 1:
        pool  = ai_train_by_model[active_generators[0]]
        texts = random.choices(pool, k=min(n, len(pool)))
        return texts, [active_generators[0]] * len(texts)

    sizes = [len(ai_train_by_model[g]) for g in active_generators]
    total = sum(sizes)
    xm_texts: List[str] = []
    xm_srcs:  List[str] = []
    for g, sz in zip(active_generators, sizes):
        quota = max(1, round(n * sz / total))
        draw  = random.choices(ai_train_by_model[g], k=min(quota, len(ai_train_by_model[g])))
        xm_texts.extend(draw)
        xm_srcs.extend([g] * len(draw))

    combined = list(zip(xm_texts, xm_srcs))
    random.shuffle(combined)
    combined = combined[:n]
    if not combined:
        return [], []
    texts_out, srcs_out = zip(*combined)
    return list(texts_out), list(srcs_out)


# ══════════════════════════════════════════════════════════════════════════════
#  Paraphraser — Gσ  (T5-large, cppo-ep)
# ══════════════════════════════════════════════════════════════════════════════
class Paraphraser(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info(f"Loading paraphraser '{PARAPHRASER_MODEL_NAME}' …")
        self.tokenizer = T5Tokenizer.from_pretrained(PARAPHRASER_MODEL_NAME)
        if PARAPHRASER_LOAD_IN_8BIT:
            # 8-bit quantization via bitsandbytes — halves VRAM (~11 GB for DIPPER-XXL).
            # device_map="auto" lets HF place layers across GPU automatically;
            # do NOT call .to(DEVICE) afterwards — device_map handles placement.
            logger.info("  Loading in 8-bit (bitsandbytes) …")
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = T5ForConditionalGeneration.from_pretrained(
                PARAPHRASER_MODEL_NAME,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                PARAPHRASER_MODEL_NAME,
                torch_dtype=torch.float16,
            )
            self.model.to(DEVICE)

    @torch.no_grad()
    def paraphrase_batch(
        self, ai_texts: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """Single-pass T5 paraphrase for PPO. Returns (texts, token_ids)."""
        model_device = next(self.model.parameters()).device
        # humarin model uses lowercase prefix; DIPPER used "Paraphrase: "
        # Both humarin and ramsrigouthamg use lowercase "paraphrase: " prefix
        prefix   = "paraphrase: " if any(x in PARAPHRASER_MODEL_NAME
                       for x in ("humarin", "ramsrigouthamg")) else "Paraphrase: "
        prefixed = [f"{prefix}{t}" for t in ai_texts]
        enc = self.tokenizer(
            prefixed, return_tensors="pt", padding=True,
            truncation=True, max_length=PARAPHRASER_MAX_INPUT_LEN,
        ).to(model_device)
        # Sanity-check model weights before generate() — NaN weights (from a failed
        # warmstart or gradient explosion) produce NaN logits → multinomial crash.
        first_param = next(self.model.parameters())
        if torch.isnan(first_param).any():
            raise RuntimeError(
                "Paraphraser weights contain NaN — model corrupted before generate(). "
                "Set WARMSTART_STEPS=0 and restart."
            )
        # Stochastic sampling with temperature=1.0 — safe (no NaN), gives reward
        # variance needed for PPO. temperature=1.5 caused the multinomial crash;
        # beam search fixed the crash but killed learning (zero reward variance).
        gen_ids = self.model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=PARAPHRASER_MAX_NEW_TOKENS,
            do_sample=True,
            top_k=PARAPHRASER_TOP_K,
            top_p=PARAPHRASER_TOP_P,
            temperature=PARAPHRASER_TEMPERATURE,
            repetition_penalty=PARAPHRASER_REPETITION_PEN,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=_NAN_SAFE_PROCESSOR,  # sanitize NaN/inf before multinomial
        )
        xp_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids
        ]
        # Clamp generated IDs to valid vocab range before storing in buffer.
        # Out-of-range IDs cause a device-side assert in compute_logprobs gather().
        vocab_size = self.model.config.vocab_size
        gen_ids    = gen_ids.clamp(min=0, max=vocab_size - 1)
        return xp_texts, gen_ids.cpu()   # always return ids on CPU for buffer storage

    def compute_logprobs(
        self, xm_texts: List[str], xp_ids: torch.Tensor
    ) -> torch.Tensor:
        """log P_{Gσ}(xp | xm) — RADAR Eq. 1."""
        model_device = next(self.model.parameters()).device
        # FIX 1: use model-aware prefix (humarin uses lowercase "paraphrase: ")
        # Both humarin and ramsrigouthamg use lowercase "paraphrase: " prefix
        prefix   = "paraphrase: " if any(x in PARAPHRASER_MODEL_NAME
                       for x in ("humarin", "ramsrigouthamg")) else "Paraphrase: "
        prefixed = [f"{prefix}{t}" for t in xm_texts]
        enc = self.tokenizer(
            prefixed, return_tensors="pt", padding=True,
            truncation=True, max_length=PARAPHRASER_MAX_INPUT_LEN,
        ).to(model_device)

        xp_ids     = xp_ids.to(model_device)
        vocab_size = self.model.config.vocab_size

        # FIX 2: clamp BOTH min and max — stray IDs >= vocab_size cause a
        # device-side assert inside gather() on the GPU.
        dec_input  = xp_ids[:, :-1].clamp(min=0, max=vocab_size - 1)
        dec_target = xp_ids[:, 1:].clamp(min=0, max=vocab_size - 1)

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=dec_input,
        )
        log_probs = F.log_softmax(out.logits, dim=-1)
        # Clamp log_probs to avoid -inf propagating into PPO loss.
        # log(softmax) can be exactly -inf for zero-probability tokens;
        # when gathered into the PPO objective this produces NaN gradients.
        log_probs = log_probs.clamp(min=-100.0)
        gathered  = log_probs.gather(2, dec_target.unsqueeze(-1)).squeeze(-1)
        # Mask out padding positions from mean (pad_token_id=0 for T5)
        mask      = (xp_ids[:, 1:] != self.tokenizer.pad_token_id).float()
        n_tokens  = mask.sum(dim=1).clamp(min=1)
        result    = (gathered * mask).sum(dim=1) / n_tokens
        # Final NaN guard — return zeros rather than NaN if something slipped through
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=-100.0)

    def warmstart_mle_loss(self, ai_texts: List[str]) -> torch.Tensor:
        """Supervised warm-start: teach T5 to copy-paraphrase before PPO."""
        model_device = next(self.model.parameters()).device
        # Both humarin and ramsrigouthamg use lowercase "paraphrase: " prefix
        prefix   = "paraphrase: " if any(x in PARAPHRASER_MODEL_NAME
                       for x in ("humarin", "ramsrigouthamg")) else "Paraphrase: "
        prefixed = [f"{prefix}{t}" for t in ai_texts]
        enc = self.tokenizer(
            prefixed, return_tensors="pt", padding=True,
            truncation=True, max_length=PARAPHRASER_MAX_INPUT_LEN,
        ).to(model_device)
        tgt_enc = self.tokenizer(
            ai_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=PARAPHRASER_MAX_INPUT_LEN,
        ).to(model_device)
        labels = tgt_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        ).loss

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Paraphraser saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Detector — Dϕ  (RoBERTa-large)
# ══════════════════════════════════════════════════════════════════════════════
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info(f"Loading detector '{DETECTOR_MODEL_NAME}' …")
        self.tokenizer = RobertaTokenizer.from_pretrained(DETECTOR_MODEL_NAME)
        self.model     = RobertaForSequenceClassification.from_pretrained(
            DETECTOR_MODEL_NAME, num_labels=2,
        )
        self.model.to(DEVICE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            self.model(input_ids=input_ids, attention_mask=attention_mask).logits,
            dim=-1,
        )[:, 1]   # P(human)

    @torch.no_grad()
    def predict_human_prob(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        model_device = next(self.model.parameters()).device
        probs: List[float] = []
        for i in range(0, len(texts), DETECTOR_BATCH):
            enc = self.tokenizer(
                texts[i : i + DETECTOR_BATCH], return_tensors="pt", padding=True,
                truncation=True, max_length=DETECTOR_MAX_LEN,
            ).to(model_device)
            probs.extend(self.forward(enc["input_ids"], enc["attention_mask"]).cpu().tolist())
        self.model.train()
        return np.array(probs)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Detector saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Loss Functions
# ══════════════════════════════════════════════════════════════════════════════
def _binary_smooth_loss(
    p: torch.Tensor,
    target_is_human: bool,
    alpha: float,
) -> torch.Tensor:
    """
    Smoothed binary cross-entropy.
    target_is_human=True  → target = (1-α), soft label for human class
    target_is_human=False → target = α,     soft label for AI class
    """
    eps = 1e-8
    if target_is_human:
        return -(
            (1 - alpha) * torch.log(p + eps)
            + alpha     * torch.log(1 - p + eps)
        ).mean()
    else:
        return -(
            (1 - alpha) * torch.log(1 - p + eps)
            + alpha     * torch.log(p + eps)
        ).mean()


def detector_multievasion_loss_and_backward(
    detector:      "Detector",
    d_opt,
    xh_texts:      List[str],
    xm_texts:      List[str],
    xp_ppo_texts:  List[str],
    xp_det_by_atk: Dict[str, List[str]],
    lambda_coeff:  float = LAMBDA,
    ls_alpha:      float = LABEL_SMOOTHING_ALPHA,
    micro_batch:   int   = ATTACK_MICRO_BATCH,
) -> Dict[str, float]:
    """
    Fix OOM-2: Memory-safe multi-attack detector loss via gradient accumulation.

    ROOT CAUSE of original OOM:
      The old version encoded ALL text groups (xh + xm + xp_ppo + 5 attacks)
      and kept all their hidden states live simultaneously so PyTorch could
      build one big computation graph before .backward().  With RoBERTa-large
      and batch=16 this consumed ~15 GB of activations on top of the weights.

    THIS VERSION:
      Processes each text group (human / original AI / T5 paraphrase /
      each attack type) in its own micro-batch forward+backward loop.
      After each group's .backward(), PyTorch frees its activation memory.
      Only ONE group's activations are live at a time (~0.5-1 GB vs ~15 GB).

      Gradient accumulation is the standard pattern: zero_grad() once before
      all groups, call .backward() per group, clip+step once after all groups.
      The optimizer step is called by the CALLER after this function returns.

    Returns: dict of scalar loss values per component (for logging only).
    """
    def encode(texts: List[str]):
        model_device = next(detector.model.parameters()).device
        enc = detector.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=DETECTOR_MAX_LEN,
        )
        return enc["input_ids"].to(model_device), enc["attention_mask"].to(model_device)

    def process_group(texts: List[str], is_human: bool, weight: float, tag: str) -> float:
        """
        Forward + immediate backward for one text group in micro-batches.
        Returns mean scalar loss for logging; tensors freed after each micro-batch.
        """
        if not texts:
            return 0.0
        total_loss_val, n_micro = 0.0, 0
        for i in range(0, len(texts), micro_batch):
            mb = texts[i : i + micro_batch]
            if not mb:
                continue
            ids, mask = encode(mb)
            p    = detector(ids, mask)
            loss = _binary_smooth_loss(p, is_human, ls_alpha)
            # Scale: human terms have weight 1; AI terms are multiplied by λ·w
            scaled = loss if is_human else lambda_coeff * weight * loss
            scaled.backward()  # accumulate into .grad; frees computation graph
            total_loss_val += loss.item()
            n_micro        += 1
            del ids, mask, p, loss, scaled  # explicit free
            torch.cuda.empty_cache()
        return total_loss_val / max(n_micro, 1)

    # Zero gradients ONCE — .backward() inside process_group accumulates into them
    d_opt.zero_grad()

    component_losses: Dict[str, float] = {}

    # Group 1: Human texts
    component_losses["L_human"]       = process_group(xh_texts,      True,  1.0,  "xh")
    # Group 2: Original AI texts
    component_losses["L_xm_original"] = process_group(xm_texts,      False, 1.0,  "xm")
    # Group 3: T5 PPO paraphrase
    if xp_ppo_texts:
        w = ATTACK_LOSS_WEIGHTS.get("t5_paraphrase", 1.0)
        component_losses["L_t5_para"] = process_group(xp_ppo_texts,  False, w,    "t5")
    # Groups 4+: Deterministic attacks — one at a time, each cleared before next
    for atk_name, atk_texts in xp_det_by_atk.items():
        if not atk_texts:
            continue
        w = ATTACK_LOSS_WEIGHTS.get(atk_name, 0.5)
        component_losses[f"L_{atk_name}"] = process_group(atk_texts, False, w, atk_name)

    component_losses["total"] = sum(component_losses.values())
    # NOTE: caller must call clip_grad_norm_ + d_opt.step() + d_sched.step()
    return component_losses


def paraphraser_cppo_ep_loss(
    paraphraser:  Paraphraser,
    xm_texts:     List[str],
    xp_ids:       torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages:   torch.Tensor,
    epsilon: float = PPO_EPSILON,
    gamma:   float = ENTROPY_COEFF,
    kl_coeff:float = KL_COEFF,
) -> torch.Tensor:
    """cppo-ep loss (RADAR Eq. 2) + KL penalty.
    Ratio clamping: exp(log_diff) can overflow to inf when weights diverge
    after a backward pass. Clamping log_diff to [-5,5] keeps ratio in [0.007, 148]
    which is finite and still gives a meaningful gradient signal.
    """
    new_logprobs = paraphraser.compute_logprobs(xm_texts, xp_ids)
    log_diff     = (new_logprobs - old_logprobs.detach().to(DEVICE)).clamp(-5.0, 5.0)
    ratio        = torch.exp(log_diff)
    adv          = advantages.detach().to(DEVICE).clamp(-3.0, 3.0)  # prevent extreme advantages
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
    L_A   = -torch.min(surr1, surr2).mean()
    L_KL  = log_diff.mean()
    L_E   = new_logprobs.mean()
    loss  = L_A + kl_coeff * L_KL + gamma * L_E
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  PPO Buffer  (now stores attack-evaded texts too)
# ══════════════════════════════════════════════════════════════════════════════
class PPOBuffer:
    def __init__(self, capacity: int = PPO_BUFFER_SIZE):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.xh_texts:      List[str]                = []
        self.xm_texts:      List[str]                = []
        self.xp_ppo_texts:  List[str]                = []   # T5 paraphrase
        self.xp_ids_list:   List[torch.Tensor]       = []
        self.old_logprobs:  List[float]              = []
        self.rewards:       List[float]              = []
        self.source_models: List[str]                = []
        # deterministic attack outputs: {attack_name: [texts]}
        self.xp_det_by_atk: Dict[str, List[str]]     = defaultdict(list)

    def add(
        self,
        xh: str, xm: str, xp_ppo: str,
        xp_id: torch.Tensor,
        old_lp: float, reward: float,
        source: str = "unknown",
        xp_det: Optional[Dict[str, str]] = None,
    ):
        self.xh_texts.append(xh)
        self.xm_texts.append(xm)
        self.xp_ppo_texts.append(xp_ppo)
        self.xp_ids_list.append(xp_id)
        self.old_logprobs.append(old_lp)
        self.rewards.append(reward)
        self.source_models.append(source)
        if xp_det:
            for atk, txt in xp_det.items():
                self.xp_det_by_atk[atk].append(txt)

    def __len__(self) -> int:
        return len(self.xh_texts)

    def compute_advantages(self) -> torch.Tensor:
        r   = torch.tensor(self.rewards, dtype=torch.float32)
        return (r - r.mean()) / (r.std() + 1e-8)

    def get_padded_ids(self, tokenizer) -> torch.Tensor:
        max_len = max(ids.shape[-1] for ids in self.xp_ids_list)
        rows = []
        for ids in self.xp_ids_list:
            pad = max_len - ids.shape[-1]
            if pad > 0:
                ids = torch.cat(
                    [ids, torch.full((1, pad), tokenizer.pad_token_id, dtype=torch.long)],
                    dim=-1,
                )
            rows.append(ids)
        return torch.cat(rows, dim=0)

    def reward_by_source(self) -> Dict[str, float]:
        totals: Dict[str, List[float]] = defaultdict(list)
        for r, s in zip(self.rewards, self.source_models):
            totals[s].append(r)
        return {k: float(np.mean(v)) for k, v in sorted(totals.items())}


# ══════════════════════════════════════════════════════════════════════════════
#  Validation — per-generator AND per-attack AUROC
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_auroc_per_generator(
    detector:        Detector,
    human_val:       List[str],
    ai_val_by_model: Dict[str, List[str]],
    outer_step:      int,
) -> Dict[str, float]:
    human_probs = detector.predict_human_prob(human_val)
    aurocs:      Dict[str, float] = {}
    lines:       List[str]        = []
    for model_name, ai_texts in sorted(ai_val_by_model.items()):
        if not ai_texts:
            continue
        ai_probs = detector.predict_human_prob(ai_texts)
        n        = min(len(human_probs), len(ai_texts))
        scores   = np.concatenate([human_probs[:n], ai_probs])
        labels   = np.concatenate([np.ones(n), np.zeros(len(ai_probs))])
        try:
            auroc = float(roc_auc_score(labels, scores))
        except ValueError:
            auroc = 0.5
        aurocs[model_name] = auroc
        lines.append(f"{outer_step}\t{model_name}\t{auroc:.4f}")

    macro = float(np.mean(list(aurocs.values()))) if aurocs else 0.5
    aurocs["MACRO_AVG"] = macro
    lines.append(f"{outer_step}\tMACRO_AVG\t{macro:.4f}")

    write_hdr = not os.path.exists(AUROC_LOG_FILE)
    with open(AUROC_LOG_FILE, "a") as f:
        if write_hdr:
            f.write("step\tgenerator\tAUROC\n")
        f.write("\n".join(lines) + "\n")
    return aurocs


def evaluate_auroc_per_attack(
    detector:        Detector,
    attack_pool:     EvasionAttackPool,
    human_val:       List[str],
    ai_val_by_model: Dict[str, List[str]],
    outer_step:      int,
) -> Dict[str, float]:
    """
    For each attack strategy, apply it to the entire AI val set and
    measure how well the detector still catches it.  Lower AUROC on an
    attack = detector is weaker against that evasion strategy.

    This diagnostic directly measures the training goal: does the detector
    remain robust when AI text is processed through each attack type?
    """
    all_ai_val = [t for texts in ai_val_by_model.values() for t in texts]
    human_probs = detector.predict_human_prob(human_val)
    aurocs: Dict[str, float] = {}
    lines:  List[str]        = []

    # Baseline: original AI text (no attack)
    ai_probs_orig = detector.predict_human_prob(all_ai_val)
    n = min(len(human_probs), len(ai_probs_orig))
    try:
        baseline_auroc = float(roc_auc_score(
            np.concatenate([np.ones(n), np.zeros(n)]),
            np.concatenate([human_probs[:n], ai_probs_orig[:n]]),
        ))
    except ValueError:
        baseline_auroc = 0.5
    aurocs["no_attack"] = baseline_auroc
    lines.append(f"{outer_step}\tno_attack\t{baseline_auroc:.4f}")

    # T5 single-pass paraphrase
    if ATTACK_T5_PARAPHRASE and attack_pool.t5_model is not None:
        xp_t5 = attack_pool.t5_paraphrase(all_ai_val[:50])  # subsample for speed
        ai_probs_t5 = detector.predict_human_prob(xp_t5)
        n2 = min(len(human_probs), len(ai_probs_t5))
        try:
            t5_auroc = float(roc_auc_score(
                np.concatenate([np.ones(n2), np.zeros(n2)]),
                np.concatenate([human_probs[:n2], ai_probs_t5[:n2]]),
            ))
        except ValueError:
            t5_auroc = 0.5
        aurocs["t5_paraphrase"] = t5_auroc
        lines.append(f"{outer_step}\tt5_paraphrase\t{t5_auroc:.4f}")

    # Deterministic attacks on ALL val AI texts
    det_attacks = attack_pool.apply_all_deterministic(all_ai_val)
    for atk_name, atk_texts in det_attacks.items():
        if not atk_texts:
            continue
        ai_probs_atk = detector.predict_human_prob(atk_texts)
        n3 = min(len(human_probs), len(ai_probs_atk))
        try:
            atk_auroc = float(roc_auc_score(
                np.concatenate([np.ones(n3), np.zeros(n3)]),
                np.concatenate([human_probs[:n3], ai_probs_atk[:n3]]),
            ))
        except ValueError:
            atk_auroc = 0.5
        aurocs[atk_name] = atk_auroc
        lines.append(f"{outer_step}\t{atk_name}\t{atk_auroc:.4f}")

    write_hdr = not os.path.exists(ATTACK_AUROC_LOG)
    with open(ATTACK_AUROC_LOG, "a") as f:
        if write_hdr:
            f.write("step\tattack\tAUROC\n")
        f.write("\n".join(lines) + "\n")

    return aurocs


def log_auroc_table(
    gen_aurocs:  Dict[str, float],
    atk_aurocs:  Dict[str, float],
    best:        float,
) -> None:
    macro = gen_aurocs.get("MACRO_AVG", 0.5)
    logger.info("  ┌─ Generator AUROC ────────────────────────────┐")
    for name, val in sorted(gen_aurocs.items()):
        flag = " ◄" if name == "MACRO_AVG" else ""
        logger.info(f"  │  {name:<26s}  {val:.4f}{flag}")
    logger.info(f"  │  (prev best)                  {best:.4f}")
    logger.info("  ├─ Attack AUROC (robustness) ─────────────────┤")
    for name, val in sorted(atk_aurocs.items()):
        delta = val - atk_aurocs.get("no_attack", val)
        arrow = "▼" if delta < -0.05 else ("▲" if delta > 0.05 else "~")
        logger.info(f"  │  {name:<26s}  {val:.4f}  {arrow}")
    logger.info("  └─────────────────────────────────────────────┘")
    if macro >= 0.999:
        logger.warning(
            "  ⚠  AUROC=1.0 — detector may be overfit / game collapsed. "
            "Raise LABEL_SMOOTHING_ALPHA or DETECTOR_UPDATE_EVERY."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════
def train_radar_multievasion():
    set_seed(SEED)

    enabled_attacks = [
        name for name, flag in [
            ("t5_paraphrase",       ATTACK_T5_PARAPHRASE),
            ("recursive_para",      ATTACK_RECURSIVE_PARA),
            ("synonym_replacement", ATTACK_SYNONYM_REPLACEMENT and NLTK_AVAILABLE),
            ("homoglyphs",          ATTACK_HOMOGLYPHS),
            ("article_deletion",    ATTACK_ARTICLE_DELETION),
            ("misspelling",         ATTACK_MISSPELLING),
        ] if flag
    ]

    logger.info("═" * 72)
    logger.info("  RADAR × RAID — Multi-Generator + Multi-Evasion")
    logger.info(f"  strategy={GENERATOR_SAMPLING_STRATEGY}  "
                f"human={NUM_HUMAN_SAMPLES}  AI={SAMPLES_PER_GENERATOR}/gen")
    logger.info(f"  Active attacks: {enabled_attacks}")
    logger.info(f"  NLTK available: {NLTK_AVAILABLE}")
    logger.info(f"  Fixes: ls={LABEL_SMOOTHING_ALPHA}  "
                f"det_every={DETECTOR_UPDATE_EVERY}  "
                f"clip=[{REWARD_CLIP_MIN},{REWARD_CLIP_MAX}]  "
                f"warmstart={WARMSTART_STEPS}  kl={KL_COEFF}")
    logger.info(f"  device={DEVICE}")
    logger.info("═" * 72)

    # ── 1. Load data ───────────────────────────────────────────────────────
    human_train, human_val, ai_train_by_model, ai_val_by_model = \
        load_raid_multigenerator()
    generators = sorted(ai_train_by_model.keys())
    if not generators:
        raise RuntimeError("No AI generator data found — check RAID filters.")

    scheduler = GeneratorScheduler(generators, GENERATOR_SAMPLING_STRATEGY)

    # ── 2. Models ──────────────────────────────────────────────────────────
    paraphraser  = Paraphraser()
    detector     = Detector()
    attack_pool  = EvasionAttackPool(
        paraphraser_model     = paraphraser.model,
        paraphraser_tokenizer = paraphraser.tokenizer,
    )
    if PARAPHRASER_LOAD_IN_8BIT:
        # 8-bit quantized models: prepare for k-bit training so gradients flow correctly.
        # This unfreezes layer norms and casts them to fp32 for stable training.
        try:
            from peft import prepare_model_for_kbit_training
            paraphraser.model = prepare_model_for_kbit_training(
                paraphraser.model, use_gradient_checkpointing=True
            )
            logger.info("  8-bit training: applied prepare_model_for_kbit_training (peft)")
        except ImportError:
            logger.warning(
                "  peft not installed — 8-bit training may be unstable.\n"
                "  Install with: pip install peft"
            )
    paraphraser.model.train()
    detector.model.train()

    # ── 3. Optimisers ──────────────────────────────────────────────────────
    total_steps = MAX_OUTER_STEPS * PPO_EPOCHS
    p_opt   = AdamW(paraphraser.parameters(), lr=PARAPHRASER_LR)
    d_opt   = AdamW(detector.parameters(),    lr=DETECTOR_LR)
    p_sched = get_linear_schedule_with_warmup(
        p_opt, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    d_sched = get_linear_schedule_with_warmup(
        d_opt, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    buffer           = PPOBuffer(capacity=PPO_BUFFER_SIZE)
    best_auroc       = 0.0
    no_improve       = 0
    detector_frozen_for = 0   # dynamic freeze: steps remaining where detector is skipped
    last_macro_auroc    = 0.0 # track AUROC for freeze decisions

    # ── 4. Warm-start (Fix 4) ──────────────────────────────────────────────
    if WARMSTART_STEPS > 0:
        logger.info(f"\n── Warm-start: {WARMSTART_STEPS} MLE steps ──")
        all_ai_ws = [t for v in ai_train_by_model.values() for t in v]
        for ws in range(1, WARMSTART_STEPS + 1):
            batch = random.choices(all_ai_ws, k=DETECTOR_BATCH)
            loss  = paraphraser.warmstart_mle_loss(batch)
            p_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(paraphraser.parameters(), GRAD_CLIP)
            p_opt.step()
            p_sched.step()
            logger.info(f"  [WS {ws:2d}/{WARMSTART_STEPS}]  MLE={loss.item():.4f}")
        logger.info("── Warm-start complete ──\n")

    # ── 5. Adversarial training loop ───────────────────────────────────────
    logger.info("Starting adversarial training loop …\n")

    for outer_step in range(1, MAX_OUTER_STEPS + 1):

        # ── 5a. Fill buffer ───────────────────────────────────────────────
        buffer.clear()
        paraphraser.model.eval()
        detector.model.eval()

        active_gens = scheduler.next(outer_step)
        n_samples   = min(PPO_BUFFER_SIZE, len(human_train))

        batch_xm, xm_sources = sample_from_generators(
            ai_train_by_model, active_gens, n_samples
        )
        batch_xh = random.choices(human_train, k=len(batch_xm))

        # Track A: T5 paraphrase → PPO token ids + rewards (T5 on GPU)
        xp_ppo_texts, xp_ids_tensor = paraphraser.paraphrase_batch(batch_xm)

        with torch.no_grad():
            old_lp_tensor = paraphraser.compute_logprobs(batch_xm, xp_ids_tensor)
        old_lp_list = old_lp_tensor.cpu().tolist()

        raw_rewards = detector.predict_human_prob(xp_ppo_texts)
        rewards     = np.clip(raw_rewards, REWARD_CLIP_MIN, REWARD_CLIP_MAX)

        # Fix OOM-4: offload T5 to CPU before running deterministic attacks.
        # Recursive paraphrase needs T5 but not its gradients.
        # Freeing T5 from GPU (~6-8 GB) before the Track-B pass prevents OOM.
        (paraphraser.model.to("cpu") if not PARAPHRASER_LOAD_IN_8BIT else None)  # 8-bit models use device_map; skip manual offload
        torch.cuda.empty_cache()  # Fix OOM-3: explicit fragmentation clear

        # Track B: deterministic attacks (T5 now on CPU for recursive_para)
        det_attacks_batch = attack_pool.apply_all_deterministic(batch_xm)

        # Reload T5 to GPU for next PPO step
        paraphraser.model.to(DEVICE)
        torch.cuda.empty_cache()

        # Store everything in the buffer
        for i in range(len(batch_xm)):
            xp_det_i = {
                atk_name: texts[i]
                for atk_name, texts in det_attacks_batch.items()
                if i < len(texts)
            }
            buffer.add(
                xh      = batch_xh[i],
                xm      = batch_xm[i],
                xp_ppo  = xp_ppo_texts[i],
                xp_id   = xp_ids_tensor[i : i + 1].cpu(),
                old_lp  = old_lp_list[i],
                reward  = float(rewards[i]),
                source  = xm_sources[i],
                xp_det  = xp_det_i,
            )

        advantages  = buffer.compute_advantages()
        all_xp_ids  = buffer.get_padded_ids(paraphraser.tokenizer)
        old_lp_buf  = torch.tensor(buffer.old_logprobs, dtype=torch.float32)
        reward_srcs = buffer.reward_by_source()

        # ── 5b. PPO update for Gσ (every step, Track A only) ─────────────
        paraphraser.model.train()
        p_loss_total, p_steps = 0.0, 0

        for _ in range(PPO_EPOCHS):
            idx_list = list(range(len(buffer)))
            random.shuffle(idx_list)
            for i in range(0, len(idx_list), DETECTOR_BATCH):
                b = idx_list[i : i + DETECTOR_BATCH]
                if not b:
                    continue
                p_loss = paraphraser_cppo_ep_loss(
                    paraphraser,
                    [buffer.xm_texts[j]     for j in b],
                    all_xp_ids[b],
                    old_lp_buf[b],
                    advantages[b],
                )
                p_loss_val = p_loss.item()
                if not (math.isfinite(p_loss_val)):
                    logger.warning(f"  PPO loss is {p_loss_val:.4f} — skipping backward to protect weights")
                    p_opt.zero_grad()
                    continue
                p_opt.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(paraphraser.parameters(), GRAD_CLIP)
                # Zero NaN/inf gradients — prevents weight explosion that causes
                # NaN logits on the next generate() call.
                for p in paraphraser.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                p_opt.step()
                p_sched.step()
                p_loss_total += p_loss_val
                p_steps      += 1

        avg_p_loss = p_loss_total / max(p_steps, 1)

        # ── 5c. Detector update (every DETECTOR_UPDATE_EVERY steps) ───────
        avg_d_loss   = float("nan")
        loss_details = {}

        # ── Dynamic freeze: skip detector when AUROC is already high ─────
        # This gives the paraphraser room to catch up before the detector
        # is allowed to train again, preventing the game from dying.
        if detector_frozen_for > 0:
            detector_frozen_for -= 1
            skip_detector = True
        elif last_macro_auroc > AUROC_FREEZE_THRESHOLD:
            detector_frozen_for = DETECTOR_FREEZE_STEPS - 1
            logger.info(
                f"  ⚠ AUROC={last_macro_auroc:.4f} > threshold={AUROC_FREEZE_THRESHOLD} "
                f"→ freezing detector for {DETECTOR_FREEZE_STEPS} steps"
            )
            skip_detector = True
        else:
            skip_detector = (outer_step % DETECTOR_UPDATE_EVERY != 0)

        if not skip_detector:
            detector.model.train()

            # Fix OOM-4: T5 off GPU during detector update — saves ~6-8 GB
            (paraphraser.model.to("cpu") if not PARAPHRASER_LOAD_IN_8BIT else None)  # 8-bit models use device_map; skip manual offload
            torch.cuda.empty_cache()

            idx_list = list(range(len(buffer)))
            random.shuffle(idx_list)
            d_loss_total, d_steps = 0.0, 0

            for i in range(0, len(idx_list), DETECTOR_BATCH):
                b = idx_list[i : i + DETECTOR_BATCH]
                if not b:
                    continue

                xp_det_batch: Dict[str, List[str]] = {
                    atk_name: [all_texts[j] for j in b if j < len(all_texts)]
                    for atk_name, all_texts in buffer.xp_det_by_atk.items()
                }

                # Fix OOM-2: gradient accumulation — zero_grad + backward
                # happen INSIDE detector_multievasion_loss_and_backward,
                # one attack group at a time. Only step+clip here.
                loss_details = detector_multievasion_loss_and_backward(
                    detector,
                    d_opt,
                    [buffer.xh_texts[j]     for j in b],
                    [buffer.xm_texts[j]     for j in b],
                    [buffer.xp_ppo_texts[j] for j in b],
                    xp_det_batch,
                )
                nn.utils.clip_grad_norm_(detector.parameters(), GRAD_CLIP)
                d_opt.step()
                d_sched.step()
                d_loss_total += loss_details.get("total", 0.0)
                d_steps      += 1

            avg_d_loss = d_loss_total / max(d_steps, 1)

            # Reload T5 for next PPO step
            paraphraser.model.to(DEVICE)
            torch.cuda.empty_cache()

        mean_reward    = float(np.mean(buffer.rewards))
        std_reward     = float(np.std(buffer.rewards))  # DIAG: low std = PPO dead zone
        raw_mean_reward = float(np.mean(raw_rewards))
        src_summary    = "  ".join(f"{g[:8]}={v:.3f}" for g, v in reward_srcs.items())
        loss_str       = "  ".join(f"{k}={v:.3f}" for k, v in loss_details.items())

        logger.info(
            f"[Step {outer_step:4d}/{MAX_OUTER_STEPS}]  "
            f"det={'skip' if np.isnan(avg_d_loss) else f'{avg_d_loss:.4f}'}  "
            f"para={avg_p_loss:.4f}  "
            f"raw_rwd={raw_mean_reward:.4f}±{std_reward:.4f}  rwd={mean_reward:.4f}"
        )
        if loss_str:
            logger.info(f"  loss components: {loss_str}")
        logger.info(f"  per-gen rewards: {src_summary}")

        # ── 5d. Validate ──────────────────────────────────────────────────
        if outer_step % VALIDATE_EVERY == 0 or outer_step == 1:
            gen_aurocs = evaluate_auroc_per_generator(
                detector, human_val, ai_val_by_model, outer_step
            )
            # Run full attack-robustness evaluation (slightly slower)
            paraphraser.model.eval()
            atk_aurocs = evaluate_auroc_per_attack(
                detector, attack_pool, human_val, ai_val_by_model, outer_step
            )
            paraphraser.model.train()

            macro = gen_aurocs["MACRO_AVG"]
            last_macro_auroc = macro   # used by dynamic freeze next step
            log_auroc_table(gen_aurocs, atk_aurocs, best_auroc)

            if GENERATOR_SAMPLING_STRATEGY == "curriculum":
                gen_only = {k: v for k, v in gen_aurocs.items() if k != "MACRO_AVG"}
                scheduler.update_curriculum_order(gen_only)

            if macro > best_auroc:
                best_auroc = macro
                no_improve = 0
                detector.save(DETECTOR_SAVE_PATH)
                paraphraser.save(PARAPHRASER_SAVE_PATH)
                logger.info(f"  ✓ New best macro AUROC = {best_auroc:.4f}  — saved.\n")
                if HF_PUSH_ENABLED and HF_PUSH_STRATEGY in ("best", "both"):
                    push_to_hub(best_auroc, trigger="best")
            else:
                no_improve += 1
                logger.info(f"  No improvement ({no_improve}/{PATIENCE})  best={best_auroc:.4f}\n")
                if no_improve >= PATIENCE:
                    logger.info("Early stopping triggered.\n")
                    break

    logger.info("═" * 72)
    logger.info(f"  Done.  Best macro AUROC = {best_auroc:.4f}")
    logger.info(f"  Detector    → {DETECTOR_SAVE_PATH}")
    logger.info(f"  Paraphraser → {PARAPHRASER_SAVE_PATH}")
    logger.info(f"  Gen AUROC   → {AUROC_LOG_FILE}")
    logger.info(f"  Atk AUROC   → {ATTACK_AUROC_LOG}")
    logger.info("═" * 72)

    # Push final checkpoint if requested
    if HF_PUSH_ENABLED and HF_PUSH_STRATEGY in ("final", "both"):
        push_to_hub(best_auroc, trigger="final")

    return best_auroc


# ══════════════════════════════════════════════════════════════════════════════
#  HuggingFace Hub — push trained models
# ══════════════════════════════════════════════════════════════════════════════
def push_to_hub(best_auroc: float, trigger: str = "best") -> None:
    """
    Push the saved detector and paraphraser checkpoints to HuggingFace Hub.

    Args:
        best_auroc : best macro AUROC achieved so far (embedded in model card)
        trigger    : "best" | "final" — logged to show which event triggered push

    Repos are created automatically if they don't already exist.
    Pushes both model weights (safetensors) and tokenizer files.
    Also uploads the AUROC log TSV files as dataset artifacts in the detector repo.

    Model card is auto-generated with training metadata so the repos are
    immediately browsable on HuggingFace with correct architecture tags.
    """
    if not HF_PUSH_ENABLED:
        return
    if HF_TOKEN == "hf_YOUR_TOKEN_HERE":
        logger.warning("HF_TOKEN not set — skipping push to hub.")
        return

    logger.info(f"\n── Pushing to HuggingFace Hub (trigger={trigger}) ──")

    try:
        # Authenticate
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi(token=HF_TOKEN)

        detector_repo_id    = f"{HF_USERNAME}/{HF_DETECTOR_REPO}"
        paraphraser_repo_id = f"{HF_USERNAME}/{HF_PARAPHRASER_REPO}"

        # ── Create repos if they don't exist ──────────────────────────────
        for repo_id, repo_type in [
            (detector_repo_id,    "model"),
            (paraphraser_repo_id, "model"),
        ]:
            try:
                api.repo_info(repo_id=repo_id, repo_type="model")
                logger.info(f"  Repo exists: {repo_id}")
            except Exception:
                api.create_repo(repo_id=repo_id, repo_type="model", private=False)
                logger.info(f"  Created repo: {repo_id}")

        # ── Model cards ───────────────────────────────────────────────────
        active_attacks = [
            name for name, flag in [
                ("t5_paraphrase",       ATTACK_T5_PARAPHRASE),
                ("recursive_para",      ATTACK_RECURSIVE_PARA),
                ("synonym_replacement", ATTACK_SYNONYM_REPLACEMENT),
                ("homoglyphs",          ATTACK_HOMOGLYPHS),
                ("article_deletion",    ATTACK_ARTICLE_DELETION),
                ("misspelling",         ATTACK_MISSPELLING),
            ] if flag
        ]

        detector_card = f"""---
language: en
license: apache-2.0
tags:
  - text-classification
  - ai-generated-text-detection
  - roberta
  - adversarial-training
metrics:
  - roc_auc
---

# RADAR Detector (RoBERTa-large)

Adversarially trained AI-generated text detector based on the RADAR framework
([Hu et al., NeurIPS 2023](https://arxiv.org/abs/2307.03838)), extended with
a multi-evasion attack pool for robust detection.

## Training

- **Base model**: `roberta-large`
- **Dataset**: [RAID](https://huggingface.co/datasets/liamdugan/raid) (Dugan et al., ACL 2024)
- **Evasion attacks seen during training**: {', '.join(active_attacks)}
- **Best macro AUROC**: {best_auroc:.4f}
- **Generators**: chatgpt, gpt2, gpt3, gpt4, cohere, cohere-chat, llama-chat,
  mistral, mistral-chat, mpt, mpt-chat

## Usage

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained("{detector_repo_id}")
model     = RobertaForSequenceClassification.from_pretrained("{detector_repo_id}")
model.eval()

text = "Your text here."
enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
print(f"P(human)={{probs[1]:.3f}}  P(AI)={{probs[0]:.3f}}")
```

## Label mapping
- Index 0 → AI-generated
- Index 1 → Human-written
"""

        paraphraser_card = f"""---
language: en
license: apache-2.0
tags:
  - text2text-generation
  - paraphrase
  - adversarial-training
  - t5
---

# RADAR Paraphraser (T5-large)

Adversarially trained paraphraser (Gσ) from the RADAR framework
([Hu et al., NeurIPS 2023](https://arxiv.org/abs/2307.03838)).
Trained via Clipped PPO with Entropy Penalty (cppo-ep) to generate
paraphrases that evade the companion RADAR detector.

## Training

- **Base model**: `t5-large`
- **Dataset**: [RAID](https://huggingface.co/datasets/liamdugan/raid)
- **Best detector macro AUROC during adversarial training**: {best_auroc:.4f}
- **Companion detector**: [{detector_repo_id}](https://huggingface.co/{detector_repo_id})

## Usage

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("{paraphraser_repo_id}")
model     = T5ForConditionalGeneration.from_pretrained("{paraphraser_repo_id}")

text    = "Paraphrase: " + "Your AI-generated text here."
inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                          top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

        # Write cards to local checkpoint dirs so they get uploaded with the model
        with open(os.path.join(DETECTOR_SAVE_PATH,    "README.md"), "w") as f:
            f.write(detector_card)
        with open(os.path.join(PARAPHRASER_SAVE_PATH, "README.md"), "w") as f:
            f.write(paraphraser_card)

        # ── Upload detector ───────────────────────────────────────────────
        logger.info(f"  Uploading detector → {detector_repo_id} …")
        api.upload_folder(
            folder_path   = DETECTOR_SAVE_PATH,
            repo_id       = detector_repo_id,
            repo_type     = "model",
            commit_message= f"RADAR detector | trigger={trigger} | AUROC={best_auroc:.4f}",
        )

        # Upload AUROC logs as extra artifacts inside the detector repo
        for log_path, remote_name in [
            (AUROC_LOG_FILE,  "per_generator_auroc.tsv"),
            (ATTACK_AUROC_LOG,"per_attack_auroc.tsv"),
            (LOG_FILE,        "training.log"),
        ]:
            if os.path.exists(log_path):
                api.upload_file(
                    path_or_fileobj = log_path,
                    path_in_repo    = f"training_logs/{remote_name}",
                    repo_id         = detector_repo_id,
                    repo_type       = "model",
                    commit_message  = f"Training logs ({trigger})",
                )

        logger.info(f"  ✓ Detector pushed → https://huggingface.co/{detector_repo_id}")

        # ── Upload paraphraser ────────────────────────────────────────────
        logger.info(f"  Uploading paraphraser → {paraphraser_repo_id} …")
        api.upload_folder(
            folder_path   = PARAPHRASER_SAVE_PATH,
            repo_id       = paraphraser_repo_id,
            repo_type     = "model",
            commit_message= f"RADAR paraphraser | trigger={trigger} | AUROC={best_auroc:.4f}",
        )
        logger.info(f"  ✓ Paraphraser pushed → https://huggingface.co/{paraphraser_repo_id}")
        logger.info("── Hub push complete ──\n")

    except Exception as e:
        # Never crash training due to a push failure
        logger.error(f"  ✗ Hub push failed ({trigger}): {e}")
        logger.error("  Training results are still saved locally — push manually if needed.")


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════
class RADARDetector:
    """
    Load saved detector and score text, with optional pre-processing by
    any of the evasion attacks (useful for adversarial robustness testing).

    Example
    -------
    >>> d = RADARDetector()
    >>> d.predict("LLMs have transformed NLP research.")
    {'p_human': 0.11, 'p_ai_generated': 0.89, 'label': 'AI-GENERATED'}
    >>> # Test with homoglyph evasion
    >>> pool = EvasionAttackPool()
    >>> evaded = pool.homoglyph_replace("LLMs have transformed NLP research.")
    >>> d.predict(evaded)
    """

    def __init__(self, model_path: str = DETECTOR_SAVE_PATH):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model     = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.eval().to(DEVICE)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        enc   = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(DEVICE)
        probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]
        return {
            "p_human":        round(float(probs[1].item()), 4),
            "p_ai_generated": round(float(probs[0].item()), 4),
            "label": "HUMAN" if probs[1] > probs[0] else "AI-GENERATED",
        }

    def benchmark_attacks(self, text: str) -> Dict[str, dict]:
        """
        Score a text under each evasion attack.
        Returns dict of attack_name → prediction dict.
        Useful for understanding which attacks the detector is most/least
        robust to on a specific example.
        """
        pool    = EvasionAttackPool()
        results = {"original": self.predict(text)}

        attacks = {
            "homoglyphs":          pool.homoglyph_replace(text),
            "article_deletion":    pool.article_deletion(text),
            "misspelling":         pool.random_misspelling(text),
        }
        if NLTK_AVAILABLE:
            attacks["synonym_replacement"] = pool.synonym_replace(text)

        for name, evaded_text in attacks.items():
            results[name] = self.predict(evaded_text)

        return results


if __name__ == "__main__":
    train_radar_multievasion()
