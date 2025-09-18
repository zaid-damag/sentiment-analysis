# src/models/esp32.py
# -*- coding: utf-8 -*-
from pathlib import Path
import torch, joblib, serial, time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to model_last2 directory (next to this script)
MODEL_DIR = Path(__file__).resolve().parent / "model_last2"

# Load tokenizer, model, and label encoder
print(f"[INFO] Loading model from: {MODEL_DIR}")
tok  = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl  = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
le   = joblib.load(MODEL_DIR / "label_encoder.pkl")["label_encoder"]
LABELS = list(le.classes_)

def predict(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    with torch.no_grad():
        inp = tok(s, return_tensors="pt", truncation=True, padding=True, max_length=128)
        pred_id = int(torch.argmax(mdl(**inp).logits))
        return LABELS[pred_id]

# Serial communication settings
PORT, BAUD = "COM10", 115200

print(f"[INFO] Opening {PORT} @ {BAUD} ...")
with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
    time.sleep(2)
    ser.reset_input_buffer()
    print("[READY] Waiting... (send 'exit' to stop)")

    while True:
        line = ser.readline().decode("utf-8", "ignore").strip()
        if not line or line.startswith(("Wi-Fi connected:", "MQTT connected:")):
            continue
        if line.lower() == "exit":
            print("[EXIT] Received 'exit'")
            break
        label = predict(line)
        print(f"[FROM PHONE] {line}\n[PREDICTED ] {label}")
        ser.write((label + "\n").encode("utf-8"))

print("[DONE] Serial closed.")
