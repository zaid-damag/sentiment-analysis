# src/models/train_model.py
import os, numpy as np, pandas as pd, torch, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

# === إعدادات قابلة للتعديل ===
MODEL_NAME   = os.getenv("MODEL_NAME", "distilbert-base-uncased")
MAX_LENGTH   = int(os.getenv("MAX_LENGTH", 128))
NUM_EPOCHS   = int(os.getenv("NUM_EPOCHS", 4))
LR_LAST2     = float(os.getenv("LR_LAST2", 2e-4))
BATCH_TRAIN  = int(os.getenv("BATCH_TRAIN", 16))
BATCH_EVAL   = int(os.getenv("BATCH_EVAL", 32))
ROOT = Path(__file__).resolve().parent   # ← هذا يشير إلى src/models/
OUTPUT_DIR = ROOT / "model_last2"

# === مسارات البيانات ===
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "src" / "data" / "processed" / "sentiment_synthetic_final.csv"
assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"

print(f"[INFO] Loading CSV: {CSV_PATH}")
df = (pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        .dropna(subset=["phrase","sentiment"]))
df = df[(df["phrase"].astype(str).str.strip()!="") &
        (df["sentiment"].astype(str).str.strip()!="")].copy()

# ترميز الليبلات
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"].astype(str))
print("[INFO] Classes:", list(le.classes_), "| counts:", dict(df["sentiment"].value_counts()))
print("[INFO] Shape:", df.shape)

# تقسيم stratify
train_df, test_df = train_test_split(
    df[["phrase","label"]],
    test_size=0.2, stratify=df["label"], random_state=42
)
tmp_dir = ROOT / "tmp_splits"
tmp_dir.mkdir(exist_ok=True, parents=True)
(train_df).to_csv(tmp_dir / "train.csv", index=False)
(test_df ).to_csv(tmp_dir / "test.csv",  index=False)
ds = load_dataset("csv", data_files={"train": str(tmp_dir/"train.csv"),
                                     "test":  str(tmp_dir/"test.csv")})

# Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tok(batch["phrase"], padding=False, truncation=True, max_length=MAX_LENGTH)
ds_tok = ds.map(tokenize, batched=True, remove_columns=["phrase"])
ds_tok.set_format(type="torch", columns=["input_ids","attention_mask","label"])

# نموذج + فك آخر طبقتين من DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(le.classes_)
)

# جمّد الكل
for p in model.base_model.parameters():
    p.requires_grad = False

# فك آخر طبقتين
if hasattr(model.base_model, "transformer"):
    for layer in model.base_model.transformer.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True

# رأس التصنيف دائماً يتدرّب
if hasattr(model, "pre_classifier"):
    for p in model.pre_classifier.parameters():
        p.requires_grad = True
for p in model.classifier.parameters():
    p.requires_grad = True

data_collator = DataCollatorWithPadding(tok)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision_w": p, "recall_w": r, "f1_w": f1}

has_cuda = torch.cuda.is_available()
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR_LAST2,
    per_device_train_batch_size=BATCH_TRAIN if has_cuda else 8,
    per_device_eval_batch_size=BATCH_EVAL if has_cuda else 16,
    gradient_accumulation_steps=1 if has_cuda else 2,
    weight_decay=0.01,
    warmup_ratio=0.1,
    do_eval=True,
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    fp16=True if has_cuda else False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["test"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
pretty = {k: (f"{v*100:.2f}%" if k!="eval_loss" else round(v,4)) for k,v in metrics.items()}
print("\n[FINAL] Metrics:", pretty)

# الحفظ
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
joblib.dump({"label_encoder": le}, Path(OUTPUT_DIR) / "label_encoder.pkl")
print(f"[SAVED] -> {OUTPUT_DIR}")
