import json
import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from raid import run_detection
from raid.utils import load_data

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Shushant/ADAL-detector-large"   # Hugging Face model
OUTPUT_FILE = "final_predictions.json"
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model from Hugging Face ──────────────────────────────────────────────
print(f"Loading detector from Hugging Face: {MODEL_NAME} on {DEVICE} ...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model     = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("Model loaded.\n")

# ── Detector function ─────────────────────────────────────────────────────────
# RAID expects P(AI-generated): higher score = more likely AI-generated
@torch.no_grad()
def my_detector(texts: list[str]) -> list[float]:
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i : i + BATCH_SIZE]
        enc    = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )

        logits = outputs.logits
        p_ai   = F.softmax(logits, dim=-1)[:, 0].cpu().tolist()  # index 0 = AI
        scores.extend(p_ai)

        print(f"  {min(i + BATCH_SIZE, len(texts)):,} / {len(texts):,}", end="\r")

    print()
    return scores

# ── Load RAID test data ───────────────────────────────────────────────────────
print("Loading RAID test split ...")
test_df = load_data(split="test", include_adversarial=True)
print(f"Loaded {len(test_df):,} rows.\n")

# ── Run detection ─────────────────────────────────────────────────────────────
print("Running detector ...")
predictions = run_detection(my_detector, test_df)
print("Done.\n")

# ── Save predictions ──────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump(predictions, f)

print(f"Predictions saved to {OUTPUT_FILE}")
print("Upload this file at https://github.com/liamdugan/raid to submit to the leaderboard.")