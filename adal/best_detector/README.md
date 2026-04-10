---
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
- **Evasion attacks seen during training**: t5_paraphrase, synonym_replacement, homoglyphs, article_deletion, misspelling
- **Best macro AUROC**: 0.9940
- **Generators**: chatgpt, gpt2, gpt3, gpt4, cohere, cohere-chat, llama-chat,
  mistral, mistral-chat, mpt, mpt-chat

## Usage

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained("Shushant/ADAL-detector-large")
model     = RobertaForSequenceClassification.from_pretrained("Shushant/ADAL-detector-large")
model.eval()

text = "Your text here."
enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
print(f"P(human)={probs[1]:.3f}  P(AI)={probs[0]:.3f}")
```

## Label mapping
- Index 0 → AI-generated
- Index 1 → Human-written
