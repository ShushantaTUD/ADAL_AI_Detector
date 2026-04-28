#!/usr/bin/env python3
"""
ADAL v3 — PAN CLEF 2026 Evaluation Script
===========================================
Loads a trained detector from HuggingFace (or a local path) and computes
the official PAN CLEF metrics on val.jsonl.

Official PAN formulas reproduced verbatim from:
  https://github.com/pan-webis-de/pan-code/blob/master/clef21/authorship-verification/pan20_verif_evaluator.py

Metrics:
  - ROC-AUC        : area under the ROC curve
  - Brier          : 1 − Brier score loss  (complement, higher = better)
  - C@1            : modified accuracy that rewards abstentions (score = 0.5)
  - F1             : binary F1 at threshold 0.5 (non-answers removed)
  - F0.5u          : precision-weighted F-measure; non-answers counted as FN
  - Mean           : arithmetic mean of the five metrics above
  - Confusion      : TN / FP / FN / TP counts at threshold 0.5

Usage
-----
    python evaluate_panclef.py \
        --model-path   <hf-repo-id-or-local-path> \
        --val-jsonl    /path/to/val.jsonl \
        --batch-size   16 \
        --max-length   512 \
        --output-json  ./evaluation_results.json

Label convention
----------------
PAN CLEF 2026 val.jsonl:   label=0 → human,  label=1 → AI-generated.
ADAL detector:             output = P(human).
PAN evaluator expects:     score → P(AI) where AI=1 is "positive".
  → We flip the detector score:  pred = 1 - P(human) = P(AI)
  → So truth=1 means AI, pred=1 means AI-prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Official PAN metrics (copied from pan_webis_de/pan-code, minor renaming)
# ═══════════════════════════════════════════════════════════════════════════
def _binarize(y: np.ndarray, threshold: float = 0.5, triple_valued: bool = False) -> np.ndarray:
    """
    Binarize scores at `threshold`. If triple_valued=True, scores exactly
    equal to the threshold are preserved as 0.5 (= "non-answer"); otherwise
    they're rounded up to 1.
    """
    y = np.array(y, dtype=np.float64)
    y = np.ma.fix_invalid(y, fill_value=threshold).filled(threshold)
    if triple_valued:
        y = np.where(y > threshold, 1.0, np.where(y < threshold, 0.0, threshold))
    else:
        y = np.where(y >= threshold, 1.0, 0.0)
    return y


def auc(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    """ROC-AUC — also considers non-answers (score = 0.5)."""
    try:
        return float(roc_auc_score(true_y, pred_y))
    except ValueError:
        return 0.0


def c_at_1(true_y: np.ndarray, pred_y: np.ndarray, threshold: float = 0.5) -> float:
    """
    C@1 (Peñas & Rodrigo 2011) — rewards predictions that leave some
    problems unanswered (exactly 0.5). Non-answers receive the average
    accuracy of the decided problems.

        C@1 = (1 / n) * (n_correct + (n_unanswered * n_correct / n))
    """
    n = float(len(pred_y))
    n_correct, n_unanswered = 0.0, 0.0
    for gt, pred in zip(true_y, pred_y):
        if pred == threshold:
            n_unanswered += 1
        elif (pred > threshold) == (gt > threshold):
            n_correct += 1.0
    if n == 0:
        return 0.0
    return (1.0 / n) * (n_correct + (n_unanswered * n_correct / n))


def f1(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    """F1 at threshold 0.5. Non-answers (score == 0.5) are removed."""
    true_f: List[float] = []
    pred_f: List[float] = []
    for gt, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_f.append(gt)
            pred_f.append(pred)
    if not pred_f:
        return 0.0
    pred_bin = _binarize(np.array(pred_f))
    try:
        return float(f1_score(true_f, pred_bin))
    except ValueError:
        return 0.0


def f_05_u_score(
    true_y: np.ndarray,
    pred_y: np.ndarray,
    pos_label: int = 1,
    threshold: float = 0.5,
) -> float:
    """
    F0.5u (Bevendorff et al. 2019). Precision-weighted F-measure where
    non-answers (score == 0.5) are counted as false negatives.

        F0.5u = (1.25 * TP) / (1.25 * TP + 0.25 * (FN + UNANSWERED) + FP)
    """
    pred_y_tri = _binarize(pred_y, triple_valued=True)
    n_tp = n_fn = n_fp = n_u = 0
    for i, pred in enumerate(pred_y_tri):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1
    denom = 1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp
    return (1.25 * n_tp) / denom if denom > 0 else 0.0


def brier(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    """
    Complement of the Brier score loss (1 − MSE) — bounded to [0, 1],
    higher = better. Considers non-answers (0.5).
    """
    try:
        return 1.0 - float(brier_score_loss(true_y, pred_y))
    except ValueError:
        return 0.0


def compute_confusion(true_y: np.ndarray, pred_y: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    """
    Build the standard binary confusion matrix at threshold 0.5.
    Non-answers (score == 0.5) are rounded UP to 1 here (same as _binarize
    with triple_valued=False) so that every prediction becomes binary.
    """
    pred_bin = _binarize(pred_y)
    # labels=[0,1] pins the matrix orientation regardless of class frequency
    tn, fp, fn, tp = confusion_matrix(true_y, pred_bin, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "TPR_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "TNR_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0,
    }


def evaluate_all(true_y: np.ndarray, pred_y: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute all PAN metrics + confusion matrix. Scores are rounded to four
    decimals for the headline metrics and three for the 'mean' — slightly
    finer than the official PAN evaluator (3 dp) since users want to see
    small deltas between checkpoints.
    """
    metrics = {
        "roc-auc": auc(true_y, pred_y),
        "brier":   brier(true_y, pred_y),
        "c@1":     c_at_1(true_y, pred_y, threshold),
        "f1":      f1(true_y, pred_y),
        "f05u":    f_05_u_score(true_y, pred_y, pos_label=1, threshold=threshold),
    }
    metrics["mean"] = float(np.mean(list(metrics.values())))
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["confusion"] = compute_confusion(true_y, pred_y, threshold)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_jsonl(path: str) -> List[Dict]:
    """Load a .jsonl file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warning] line {ln}: {e}", file=sys.stderr)
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Detector Inference
# ═══════════════════════════════════════════════════════════════════════════
class ADALDetector:
    """
    Wrapper around the ADAL RoBERTa-large detector. The model outputs two
    logits; the detector was trained with index 0 = human, index 1 = AI.

    Returns P(AI) so that the score matches PAN's positive-class convention
    (AI-generated = 1 = "positive detection").
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
    ) -> None:
        print(f"Loading detector from: {model_path}")
        print(f"Device: {device}")
        self.device     = device
        self.max_length = max_length
        self.tokenizer  = AutoTokenizer.from_pretrained(model_path)
        self.model      = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device).eval()
        print(f"  Loaded. num_labels={self.model.config.num_labels}  "
              f"hidden_size={self.model.config.hidden_size}")

    @torch.inference_mode()
    def predict_p_ai(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Return P(AI) ∈ [0, 1] for each input text.

        Label convention (matches ADAL training):
          - model logit index 0 → P(human)
          - model logit index 1 → P(AI)
        """
        probs = []
        n = len(texts)
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1)
            # P(AI) = P(label=1)
            # probs.append(p[:, 1].cpu().numpy())
            probs.append((1.0 - p[:, 1]).cpu().numpy())
            if (start // batch_size) % 20 == 0:
                print(f"  inference: {min(start + batch_size, n):6d} / {n}")
        return np.concatenate(probs, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
#  Optional Isotonic Calibration
# ═══════════════════════════════════════════════════════════════════════════
def try_load_calibrator(model_path: str):
    """
    If the trained checkpoint bundled an isotonic_calibrator.pkl (saved by
    ADAL v3 training at end of run), load and return it. Returns None if
    the file is missing or can't be loaded.

    Note on label direction: the training-time calibrator was fit on
    P(human) → human_label. To apply it on our P(AI) scores we must flip.
    """
    cal_path = os.path.join(model_path, "isotonic_calibrator.pkl")
    if not os.path.exists(cal_path):
        return None
    try:
        import pickle
        with open(cal_path, "rb") as f:
            cal = pickle.load(f)
        print(f"  Found isotonic calibrator: {cal_path}")
        return cal
    except Exception as e:
        print(f"  [warning] isotonic calibrator load failed: {e}")
        return None


def apply_calibration(p_ai: np.ndarray, calibrator) -> np.ndarray:
    """
    Apply training-time isotonic calibrator (fit on P(human)) to P(AI) scores.

    The calibrator maps raw_P_human → calibrated_P_human.
    To calibrate P(AI), we calibrate the equivalent P(human) = 1 − P(AI),
    then flip the result back to P(AI).
    """
    p_human_raw = 1.0 - p_ai
    p_human_cal = calibrator.predict(p_human_raw)
    p_human_cal = np.clip(p_human_cal, 0.0, 1.0)
    return 1.0 - p_human_cal


# ═══════════════════════════════════════════════════════════════════════════
#  Pretty-printing
# ═══════════════════════════════════════════════════════════════════════════
def print_results(title: str, results: Dict) -> None:
    print()
    print("═" * 72)
    print(f"  {title}")
    print("═" * 72)
    headline_keys = ["roc-auc", "brier", "c@1", "f1", "f05u", "mean"]
    for k in headline_keys:
        if k in results:
            print(f"  {k:<12s}  {results[k]:.4f}")
    print()
    print("  Confusion matrix (threshold = 0.5):")
    cm = results["confusion"]
    print(f"                 Predicted: HUMAN   Predicted: AI")
    print(f"    Actual: HUMAN  {cm['TN']:>12d}   {cm['FP']:>12d}    ← FP matters for academic use")
    print(f"    Actual: AI     {cm['FN']:>12d}   {cm['TP']:>12d}")
    print()
    print(f"  Accuracy                 {cm['accuracy']:.4f}")
    print(f"  TPR (sensitivity / recall) {cm['TPR_sensitivity']:.4f}")
    print(f"  TNR (specificity)          {cm['TNR_specificity']:.4f}")
    print(f"  FPR (false positive rate)  {cm['FPR']:.4f}   ← lower is better")
    print(f"  FNR (false negative rate)  {cm['FNR']:.4f}")
    print()


def print_per_generator(
    pred_p_ai: np.ndarray,
    records: List[Dict],
) -> None:
    """Per-generator accuracy breakdown (AI-side only, humans pool together)."""
    groups: Dict[str, List[Tuple[int, float]]] = {}
    for rec, score in zip(records, pred_p_ai):
        model_name = str(rec.get("model", "unknown")).strip()
        label      = int(rec.get("label", -1))
        key = "human" if label == 0 else model_name
        groups.setdefault(key, []).append((label, float(score)))

    print("  Per-source breakdown:")
    print(f"    {'source':<32s}  {'n':>6s}  {'mean P(AI)':>10s}  {'correct@0.5':>12s}")
    for key, items in sorted(groups.items()):
        n = len(items)
        mean_p = np.mean([s for _, s in items])
        n_correct = sum(1 for lbl, s in items if (lbl == 1 and s >= 0.5) or (lbl == 0 and s < 0.5))
        print(f"    {key:<32s}  {n:>6d}  {mean_p:>10.4f}  {n_correct:>6d} / {n:<4d}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ADAL v3 — PAN CLEF evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Shushant/ADAL-detector-large",
        help="HuggingFace repo id OR local path to detector checkpoint",
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        default="/home/shushanta/ADAL_AI_Detector/val.jsonl",
        help="Path to val.jsonl with {text, label, model} fields",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--output-json",
        type=str,
        default="./evaluation_results.json",
        help="Where to write the final metrics JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Don't apply isotonic calibrator even if it's bundled",
    )
    args = parser.parse_args()

    print("═" * 72)
    print("  ADAL v3 — PAN CLEF 2026 Evaluation")
    print("═" * 72)
    print(f"  Model       : {args.model_path}")
    print(f"  Val file    : {args.val_jsonl}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max length  : {args.max_length}")
    print(f"  Device      : {args.device}")
    print(f"  Output JSON : {args.output_json}")
    print("═" * 72)

    # ── 1. Load val data ──────────────────────────────────────────────────
    if not os.path.exists(args.val_jsonl):
        sys.exit(f"ERROR: val.jsonl not found: {args.val_jsonl}")

    print("\n[1/3] Loading val.jsonl …")
    records = load_jsonl(args.val_jsonl)
    texts   = [str(r.get("text", "")).strip() for r in records]
    labels  = np.array([int(r.get("label", -1)) for r in records], dtype=np.float64)

    # Keep only rows with valid label ∈ {0, 1}
    keep = (labels == 0) | (labels == 1)
    if not keep.all():
        n_dropped = int((~keep).sum())
        print(f"  [warning] dropping {n_dropped} rows with invalid labels")
        records = [r for r, k in zip(records, keep) if k]
        texts   = [t for t, k in zip(texts, keep) if k]
        labels  = labels[keep]

    n_human = int((labels == 0).sum())
    n_ai    = int((labels == 1).sum())
    print(f"  Loaded {len(texts):,} rows   human={n_human:,}  AI={n_ai:,}")
    if n_human == 0 or n_ai == 0:
        sys.exit("ERROR: val set has only one class — metrics will be undefined.")

    # ── 2. Load model & run inference ─────────────────────────────────────
    print("\n[2/3] Loading detector & running inference …")
    detector = ADALDetector(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
    )
    pred_p_ai = detector.predict_p_ai(texts, batch_size=args.batch_size)
    assert len(pred_p_ai) == len(labels), "Score/label count mismatch"

    # ── 3. Compute metrics (raw + optionally calibrated) ──────────────────
    print("\n[3/3] Computing PAN metrics …")

    raw_results = evaluate_all(labels, pred_p_ai)
    print_results("RAW scores (no calibration)", raw_results)
    print_per_generator(pred_p_ai, records)

    calibrated_results: Optional[Dict] = None
    if not args.skip_calibration:
        calibrator = try_load_calibrator(args.model_path)
        if calibrator is not None:
            pred_p_ai_cal = apply_calibration(pred_p_ai, calibrator)
            calibrated_results = evaluate_all(labels, pred_p_ai_cal)
            print_results("CALIBRATED scores (isotonic regression)", calibrated_results)

            # Show the lift
            print("  Calibration lift (cal − raw):")
            for k in ["roc-auc", "brier", "c@1", "f1", "f05u", "mean"]:
                d = calibrated_results[k] - raw_results[k]
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
                print(f"    {k:<12s} {d:+.4f}  {arrow}")
            print()

    # ── 4. Write JSON output ──────────────────────────────────────────────
    out = {
        "model_path": args.model_path,
        "val_file":   args.val_jsonl,
        "n_samples":  int(len(labels)),
        "n_human":    n_human,
        "n_ai":       n_ai,
        "raw":        raw_results,
    }
    if calibrated_results is not None:
        out["calibrated"] = calibrated_results

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✓ Results written to {args.output_json}")
    print("═" * 72)


if __name__ == "__main__":
    main()