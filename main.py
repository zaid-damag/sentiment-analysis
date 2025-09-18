# main.py
# -*- coding: utf-8 -*-
from pathlib import Path
import torch, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Paths ===
MODEL_DIR = Path(__file__).resolve().parent / "src" / "models" / "model_last2"

# === Load model ===
print(f"[INFO] Loading model from: {MODEL_DIR}")
tok  = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl  = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
le   = joblib.load(MODEL_DIR / "label_encoder.pkl")["label_encoder"]
LABELS = list(le.classes_)

def predict_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown"
    with torch.no_grad():
        inp = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        pred_id = int(torch.argmax(mdl(**inp).logits))
        return LABELS[pred_id]

# === CLI loop ===
if __name__ == "__main__":
    print("=== Sentiment Classifier (main.py) ===")
    print("Type any text to classify, or 'exit' to quit.\n")

    while True:
        text = input("📝 Text: ").strip()
        if text.lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break
        if not text:
            continue
        label = predict_text(text)
        print(f"{text}  -->  {label}\n")
