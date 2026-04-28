---
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
- **Best detector macro AUROC during adversarial training**: 0.9957
- **Companion detector**: [Shushant/adal-v3-panclef](https://huggingface.co/Shushant/adal-v3-panclef)

## Usage

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Shushant/adal-v3-t5-paraphraser")
model     = T5ForConditionalGeneration.from_pretrained("Shushant/adal-v3-t5-paraphraser")

text    = "Paraphrase: " + "Your AI-generated text here."
inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                          top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
