
# ADAL: AI-Generated Text Detection using Adversarial Learning
AI-generated text detection algorithm based on Adaptive Learning and Adversarial Training


**Shushanta Pudasaini** · TU Dublin
PhD Research · September 2023 – September 2027

---

## Overview

ADAL is an adversarially trained AI-generated text detector based on the RADAR framework (Hu et al., NeurIPS 2023), extended to the RAID benchmark with multi-generator training and a multi-evasion attack pool. The system trains a detector (RoBERTa-large) and a paraphraser (T5-base) in an adversarial game: the paraphraser learns to rewrite AI-generated text so it evades detection, while the detector learns to remain robust against those rewrites. The result is a detector that generalises across 11 AI generators and maintains high AUROC under five distinct evasion attacks.

Best result: **macro AUROC 0.9940** across all 11 RAID generators, robust to all attack types.

---

## Background

The proliferation of large language models has made AI-generated text increasingly difficult to distinguish from human writing. Existing detectors tend to be brittle — they perform well on clean text but degrade significantly when the text is lightly paraphrased or subjected to simple character-level perturbations. ADAL addresses this by training the detector adversarially: rather than optimising purely for clean-text detection, the detector is hardened against a pool of evasion strategies applied during training.

The approach draws on two complementary resources:

- **RADAR** (Robust AI-Text Detection via Adversarial Learning) — the adversarial training framework using Clipped PPO with Entropy Penalty (cppo-ep) to train the paraphraser against the detector.
- **RAID** (Robust AI-text Detection benchmark) — a large-scale dataset covering 11 generators, 4 decoding strategies, and 12 attack types across 11 domains.

---

## Architecture

```
RAID train split (attack='none')
        │
        ▼
   ┌────────────┐      ┌─────────────────────────────────┐
   │  xm (AI)   │─────▶│  Gσ — Paraphraser (T5-base)     │──▶ xp_ppo
   └────────────┘      │  ramsrigouthamg/t5_paraphraser  │
                       └─────────────────────────────────┘
                                        │
                              PPO reward R(xp, φ)
                                        │
   ┌────────────┐      ┌─────────────────────────────────┐
   │  xh (human)│─────▶│  Dϕ — Detector (RoBERTa-large)  │──▶ AUROC
   │  xm (AI)   │─────▶│  roberta-large                  │
   │  xp_ppo    │─────▶│  (trained via reweighted        │
   │  xp_det_k  │─────▶│   logistic loss)                │
   └────────────┘      └─────────────────────────────────┘
```

The detector is trained on human text, original AI text, T5-paraphrased AI text, and four deterministic attack variants simultaneously. The paraphraser is updated via PPO to maximise the reward signal (the detector's human-probability score assigned to its paraphrases).

---

## Dataset

All experiments use the **RAID dataset** (`liamdugan/raid`, ACL 2024).

| Split | Human texts | AI generators | AI texts per generator | Attack filter |
|---|---|---|---|---|
| Train | ~13,364 | 11 | ~26,000–53,000 | `attack='none'` (clean only) |
| Val (internal 10%) | ~1,364 | 11 | ~2,600–5,300 | same |

**AI generators:** chatgpt, gpt-3, gpt-4, gpt-2, cohere, cohere-chat, llama-chat, mistral, mistral-chat, mpt, mpt-chat

---

## Attack Pool

Training uses a two-track multi-evasion attack pool:

| Track | Attack | Description |
|---|---|---|
| A — Learnable | T5 paraphrase | PPO-trained; provides the reward signal to the paraphraser |
| B — Deterministic | Synonym replacement | WordNet POS-aware; 20% token replacement rate |
| B — Deterministic | Homoglyphs | ASCII → Unicode substitution; 10% character rate |
| B — Deterministic | Article deletion | Removes a / an / the; 50% drop rate |
| B — Deterministic | Misspelling | QWERTY adjacency typos; 8% character rate |

The detector's training loss is extended across all five attack types:

```
L_D(φ) = L_human + λ·(L_xm + L_t5_para + L_synonym + L_homoglyphs + L_article + L_misspelling)
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Paraphraser | `ramsrigouthamg/t5_paraphraser` (T5-base, 250M) |
| Detector | `roberta-large` (355M) |
| Paraphraser LR | 2e-5 |
| Detector LR | 3e-6 |
| PPO epsilon (clip) | 0.2 |
| PPO epochs per buffer | 8 |
| PPO buffer size | 64 |
| KL coefficient | 0.001 |
| Label smoothing α | 0.15 |
| Detector update frequency | every 2 outer steps |
| AUROC freeze threshold | 0.995 |
| Max outer steps | 200 |
| Early stopping patience | 25 steps |
| Generation | `do_sample=True`, `top_k=50`, `temperature=1.0` |

**Training stability fixes applied:**
- `NanSafeLogitsProcessor` — intercepts NaN/inf logits before `torch.multinomial` inside every `generate()` call
- PPO log-ratio clamped to `[-5, 5]` to prevent `exp()` overflow
- Advantage clamped to `[-3, 3]`
- NaN gradients zeroed after PPO backward before `optimizer.step()`
- `compute_logprobs` output clamped at −100 and passed through `nan_to_num`

---

## Results

### Per-Generator AUROC (validation set, best checkpoint)

| Generator | AUROC |
|---|---|
| gpt-4 | 0.9995 |
| llama-chat | 0.9994 |
| mistral-chat | 0.9994 |
| chatgpt | 0.9991 |
| mpt-chat | 0.9991 |
| gpt-3 | 0.9982 |
| gpt-2 | 0.9954 |
| cohere-chat | 0.9934 |
| mistral | 0.9913 |
| mpt | 0.9865 |
| cohere | 0.9852 |
| **Macro average** | **0.9951** |

### Per-Attack AUROC (robustness evaluation)

| Attack | AUROC |
|---|---|
| T5 paraphrase | 1.0000 |
| Misspelling | 0.9996 |
| Homoglyphs | 0.9994 |
| No attack | 0.9994 |
| Article deletion | 0.9993 |
| Synonym replacement | 0.9990 |

All attacks report `~` (no significant degradation vs the no-attack baseline).

---

## Repository Structure

```
adal/
├── fourth_adal_train.py           # Main adversarial training script
├── submit_to_raid_leaderboard.py  # RAID leaderboard submission script
├── radar_multievasion/
│   ├── best_detector/             # Saved RoBERTa-large detector checkpoint
│   ├── best_paraphraser/          # Saved T5-base paraphraser checkpoint
│   ├── per_generator_auroc.tsv    # Per-generator AUROC at best checkpoint
│   └── per_attack_auroc.tsv       # Per-attack AUROC at best checkpoint
└── FOURTH_ADAL.log                # Full training log
```

---

## Installation

```bash
pip install raid-bench torch transformers scikit-learn nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')"
```

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python fourth_adal_train.py > FOURTH_ADAL.log 2>&1
```

Key config variables at the top of `fourth_adal_train.py`:

```python
NUM_HUMAN_SAMPLES     = None   # None = all available (~13,364)
SAMPLES_PER_GENERATOR = None   # None = all available (~26k–53k per generator)
VAL_SPLIT_RATIO       = 0.1    # 10% held out for validation
MAX_OUTER_STEPS       = 200
PATIENCE              = 25     # early stopping
```

---



## References

```bibtex
@inproceedings{hu2023radar,
  title     = {RADAR: Robust AI-Text Detection via Adversarial Learning},
  author    = {Hu, Xiaomeng and Chen, Pin-Yu and Ho, Tsung-Yi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{dugan2024raid,
  title     = {RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors},
  author    = {Dugan, Liam and Hwang, Alyssa and Trhlik, Filip and Ippolito, Daphne and Callison-Burch, Chris},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}
```

---

## Author

**Shushanta Pudasaini**  
PhD Researcher, Technological University Dublin
Supervisors: Dr. Marisa Llorens Salvador · Dr. Luis Miralles-Pechuán · Dr. David Lillis
