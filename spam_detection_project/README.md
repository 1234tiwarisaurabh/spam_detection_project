# 🛡️ Spam Shield AI — Spam Detection Project

Full-stack ML web app: **Flask + Scikit-learn (TF-IDF + Multinomial Naive Bayes)** backend,
animated glassmorphism frontend. Ready to run in VS Code.

## Project Structure
```
spam_detection_project/
├── app.py                 # Flask backend (serves UI + /predict API)
├── train_model.py         # Builds dataset, trains model, saves model.pkl
├── requirements.txt
├── data/spam.csv           # generated after training
├── model/                  # generated after training (model.pkl, vectorizer.pkl, metrics.pkl)
├── templates/index.html    # animated UI
└── static/
    ├── style.css            # glassmorphism + animations
    └── script.js            # fetch API calls + bar/particle animations
```

## Setup in VS Code (Windows/Mac/Linux)

1. Open this folder in VS Code: `File > Open Folder`
2. Open a terminal in VS Code: `` Ctrl + ` ``
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Train the model (generates model.pkl, vectorizer.pkl, and the dataset):
   ```bash
   python train_model.py
   ```
6. Run the app:
   ```bash
   python app.py
   ```
7. Open your browser at **http://127.0.0.1:5000**

## How It Works
- `train_model.py` generates a realistic synthetic dataset of spam/ham messages,
  vectorizes text using **TF-IDF (unigrams + bigrams)**, and trains a
  **Multinomial Naive Bayes** classifier — the standard, proven approach for text spam
  detection (same family of algorithm used in classic SMS spam filters).
- `app.py` loads the trained model and exposes a `/predict` JSON endpoint.
- The frontend sends the typed message to `/predict`, then animates the
  spam/ham probability bars and result card based on the response.

## Want to use a REAL dataset instead?
Replace the dataset generation in `train_model.py` with the classic **SMS Spam Collection
Dataset** (5,574 labeled messages) — download it, save as `data/spam.csv` with columns
`label,text`, then skip the `build_dataset()` call and load your CSV directly with
`pd.read_csv("data/spam.csv")`. Everything else (vectorizer, model, app) works unchanged.

## For Your College Report / Viva
- **Algorithm**: Multinomial Naive Bayes (probabilistic, works well on word-frequency text data)
- **Feature extraction**: TF-IDF with unigrams + bigrams, max 3000 features
- **Metrics reported**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Why Naive Bayes**: Fast, works well with small/medium text datasets, industry-standard
  baseline for spam filtering (Gmail's early spam filter used similar principles)
- **Possible extensions to mention in viva**: try SVM/Logistic Regression for comparison,
  use real SMS Spam Collection dataset, add word-cloud visualization, deploy on Render/Railway

## Troubleshooting
- `ModuleNotFoundError` → run `pip install -r requirements.txt` again inside your venv
- `Model files not found` error when running `app.py` → run `python train_model.py` first
- Port already in use → change `app.run(debug=True)` to `app.run(debug=True, port=5001)`
