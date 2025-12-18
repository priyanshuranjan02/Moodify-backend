from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import csv, io, os

# ================= FIREBASE SETUP =================
import firebase_admin
from firebase_admin import credentials, firestore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "moodify-firebase-key.json")

cred = credentials.Certificate(FIREBASE_KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)

# ================= MODEL CONFIG =================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = None
model = None

# Reduce CPU & memory usage (IMPORTANT for Render free tier)
torch.set_num_threads(1)

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print("🔥 DistilBERT model loaded")

# ================= SENTIMENT LOGIC =================
def predict_sentiment(text: str):
    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    predicted = torch.argmax(probs).item()
    confidence = float(probs[0][predicted].item())

    sentiment = "Positive" if predicted == 1 else "Negative"

    # Neutral logic (confidence-based)
    if confidence < 0.60:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }

# ================= TEXT ANALYSIS =================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = predict_sentiment(text)

    # Store in Firebase
    db.collection("sentiments").add({
        "text": text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "timestamp": datetime.utcnow(),
        "source": "text"
    })

    return jsonify(result)

# ================= CSV ANALYSIS =================
@app.route("/predict/csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    content = file.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))

    headers = [h.lower() for h in reader.fieldnames]
    text_col = next(
        (c for c in ["review", "text", "content", "comment"] if c in headers),
        None
    )

    if not text_col:
        return jsonify({"error": "No review/text column found"}), 400

    results = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "total": 0
    }

    for row in reader:
        text = row.get(text_col, "").strip()
        if not text:
            continue

        pred = predict_sentiment(text)

        db.collection("sentiments").add({
            "text": text,
            "sentiment": pred["sentiment"],
            "confidence": pred["confidence"],
            "timestamp": datetime.utcnow(),
            "source": "csv"
        })

        results["total"] += 1
        results[pred["sentiment"].lower()] += 1

    return jsonify(results)

# ================= HISTORY =================
@app.route("/history", methods=["GET"])
def history():
    docs = (
        db.collection("sentiments")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(200)
        .stream()
    )

    data = []
    for doc in docs:
        d = doc.to_dict()
        data.append({
            "text": d.get("text"),
            "sentiment": d.get("sentiment"),
            "confidence": d.get("confidence"),
            "timestamp": d.get("timestamp").isoformat()
        })

    return jsonify(data)

# ================= STATS =================
@app.route("/stats", methods=["GET"])
def stats():
    docs = db.collection("sentiments").stream()

    stats = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "total": 0
    }

    for doc in docs:
        s = doc.to_dict().get("sentiment", "").lower()
        if s in stats:
            stats[s] += 1
        stats["total"] += 1

    return jsonify(stats)

# ================= ROOT =================
@app.route("/")
def home():
    return "🔥 Moodify Backend (DistilBERT + Firebase) Running!"

# ================= RUN =================
if __name__ == "__main__":
    app.run(port=5000)
