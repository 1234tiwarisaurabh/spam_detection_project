"""
app.py
Flask backend for the Spam Detection web app.

Run: python app.py
Then open http://127.0.0.1:5000
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pkl")
VEC_PATH = os.path.join("model", "vectorizer.pkl")
METRICS_PATH = os.path.join("model", "metrics.pkl")

model = None
vectorizer = None
metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}


def load_artifacts():
    global model, vectorizer, metrics
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        raise FileNotFoundError(
            "Model files not found. Run 'python train_model.py' first to train the model."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()

    if not text:
        return jsonify({"error": "Please enter a message to analyze."}), 400

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")

    confidence = float(proba[spam_idx] if pred == "spam" else proba[ham_idx])

    return jsonify({
        "label": pred,
        "confidence": round(confidence * 100, 2),
        "spam_probability": round(float(proba[spam_idx]) * 100, 2),
        "ham_probability": round(float(proba[ham_idx]) * 100, 2),
    })


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
