# src/data/clean_null.py
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

# هذا الملف نفسه موجود في src/data، فنحسب المسار نسبةً لمكانه
CURRENT_DIR    = Path(__file__).resolve().parent        # ...\src\data
RAW_DIR        = CURRENT_DIR / "raw"                   # ...\src\data\raw
PROCESSED_DIR  = CURRENT_DIR / "processed"             # ...\src\data\processed

RAW_FILE       = RAW_DIR / "sentiment_synthetic.csv"
PROCESSED_FILE = PROCESSED_DIR / "sentiment_synthetic_final.csv"

def clean_data():
    print("🔎 RAW_FILE :", RAW_FILE)
    print("📂 OUTPUT   :", PROCESSED_FILE)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"الملف غير موجود: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")

    # تنظيف
    df = df.dropna(subset=['phrase', 'sentiment'])
    df = df[(df['phrase'].astype(str).str.strip() != "") & 
            (df['sentiment'].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8-sig")

    print(f"✅ عدد الصفوف بعد التنظيف: {len(df)}")
    print(f"📄 الملف النظيف محفوظ في: {PROCESSED_FILE.resolve()}")

if __name__ == "__main__":
    clean_data()
