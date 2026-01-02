
                 sentiment-analysis





### Download / Clone from GitHub

To get the project from GitHub:

```bash
git clone https://github.com/ali-771/sentiment-analysis.git
cd sentiment-analysis
```

Then install dependencies using `uv` and setup the environment:

```bash
uv install
```

(Assumes `pyproject.toml` includes all dependencies like transformers, pyserial, sklearn, etc.)

---


```markdown
# Sentiment Analysis Project

This project implements a **Sentiment Analysis System** using Hugging Face Transformers (DistilBERT).  
The model is fine-tuned on a prepared dataset and can be used both for **direct text predictions** and for **integration with ESP32** to receive messages and classify them in real-time.

---

## 📂 Project Structure

```

New\_folder/
├─ src/
│  ├─ data/
│  │   ├─ raw/               # raw dataset
│  │   ├─ processed/         # cleaned dataset
│  │   └─ clean\_null.py      # preprocessing script
│  └─ models/
│      ├─ train\_model.py     # training script
│      ├─ esp32.py           # ESP32 bridge (serial communication)
│      ├─ model\_last2/       # trained model
│      └─ compact\_sentiment\_predict.py
├─ main.py                   # main entry point for CLI predictions
├─ pyproject.toml            # project dependencies (managed by uv)
└─ uv.lock

````

---

## 📥 Download / Clone from GitHub

```bash
git clone https://github.com/ali-771/sentiment-analysis.git
cd sentiment-analysis
````

---

## 🚀 How to Run

### 1. Data Cleaning

```bash
uv run python src/data/clean_null.py
```

* Loads the raw dataset (`sentiment_synthetic.csv`).
* Removes null/empty rows.
* Saves the cleaned dataset as `sentiment_synthetic_final.csv`.

### 2. Model Training

```bash
uv run python src/models/train_model.py
```

* Fine-tunes **DistilBERT** on the processed dataset.
* Saves the trained model into `src/models/model_last2/`.

### 3. Prediction (CLI)

```bash
uv run python main.py
```

Example:

```
📝 Text: I love this
I love this  -->  positive
```

### 4. ESP32 Bridge

```bash
uv run python src/models/esp32.py
```

* Opens a serial connection to ESP32 (default `COM10`, `115200 baud`).
* Reads messages from ESP32.
* Predicts sentiment using the trained model.
* Sends the prediction back to ESP32.

---

## 👥 Engineers:

* **Ali Al-Khaledi** – 202270295
* **Zaid Damag** – 202170119

