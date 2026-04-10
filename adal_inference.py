from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained("Shushant/ADAL_AI_Detector")
model     = RobertaForSequenceClassification.from_pretrained("Shushant/ADAL_AI_Detector")
model.eval()

text = """
The widespread adoption of Large Language Models (LLMs) has made the detection of AI-generated
text a pressing and complex challenge. Although many detection systems report high benchmark
accuracy, their reliability in real-world settings remains uncertain, and their interpretability is often
unexplored. In this work, we investigate whether contemporary detectors genuinely identify machine
authorship or merely exploit dataset-specific artefacts. We propose an interpretable detection framework that integrates linguistic feature engineering, machine learning, and explainable AI techniques.
Evaluated across two major benchmark corpora—PAN-CLEF 2025 and COLING 2025—models
trained on 38 linguistic features achieve leaderboard-competitive performance, attaining an F1 score
of 0.9734 without reliance on large-scale language models.
However, systematic cross-domain and cross-generator evaluation reveals substantial generalisation
failure: classifiers that excel in-domain degrade significantly under distribution shift. Using SHAPbased explanations, we show that the most influential features differ markedly between datasets,
indicating that detectors often rely on dataset-specific stylistic cues rather than stable signals of
machine authorship. Further investigating this, we perform an in-depth error analysis, applying SHAP
to False Negatives and False Positives from the model.
Our findings show that benchmark accuracy is not reliable evidence of authorship detection: strong
in-domain performance can coincide with substantial failure under domain and generator shift. Our
in-depth error analysis exposes a fundamental tension in linguistic-feature-based AI text detection:
the features that are most discriminative on in-domain data are also the features most susceptible to
domain shift, formatting variation, and text-length effects. We believe that this knowledge helps in
building AI detectors that are robust under different settings. To support replication and practical
use, we release an open-source
"""
enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
print(f"P(human)={probs[1]:.3f}  P(AI)={probs[0]:.3f}")
