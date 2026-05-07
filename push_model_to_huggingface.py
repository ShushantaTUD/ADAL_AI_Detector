import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import HfApi, create_repo

# ── Config ─────────────────────────────────────────────────────────────
MODEL_DIR = "/home/shushanta/ADAL_AI_Detector/adal_v4_panclef/best_detector"
REPO_NAME = "ADAL_AI_Detector_v2_panclef"   # change this
HF_USERNAME = "Shushant"    # change this

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ─────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_DIR}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── Create repo on Hugging Face ────────────────────────────────────────
repo_id = f"{HF_USERNAME}/{REPO_NAME}"

print(f"Creating repo: {repo_id}")
create_repo(repo_id, exist_ok=True)

# ── Push to Hugging Face ───────────────────────────────────────────────
print("Pushing model...")
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("✅ Model successfully uploaded!")
print(f"https://huggingface.co/{repo_id}")