from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS
from datetime import datetime
import csv, io, os

# ====================== FIREBASE SETUP ======================
import firebase_admin
from firebase_admin import credentials, firestore

# IMPORTANT:
# Run app.py from backend folder
# File should be located at: backend/moodify-firebase-key.json
cred = credentials.Certificate("moodify-firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ====================== FLASK APP ==========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ====================== LOAD DISTILBERT MODEL =========================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

print("🔥 DistilBERT model loaded successfully!")

# ====================== PREDICTION LOGIC ====================
def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    predicted = torch.argmax(probs).item()
    confidence = float(probs[0][predicted].item())

    label_map = {
        0: "Negative",
        1: "Positive"
    }

    return {
        "sentiment": label_map[predicted],
        "confidence": round(confidence, 4)
    }

# ====================== SINGLE TEXT ====================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = predict_sentiment(text)

    # Save to Firebase
    db.collection("sentiments").add({
        "text": text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "timestamp": datetime.utcnow(),
        "source": "text"
    })

    return jsonify(result)

# ====================== CSV ANALYSIS ====================
@app.route("/predict/csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    content = file.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))

    headers = [h.lower() for h in reader.fieldnames]
    text_col = None

    for col in ["review", "reviews", "text", "content", "comment", "message"]:
        if col in headers:
            text_col = col
            break

    if not text_col:
        return jsonify({"error": "No review/text column found"}), 400

    results = {
        "positive": 0,
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
        if pred["sentiment"] == "Positive":
            results["positive"] += 1
        else:
            results["negative"] += 1

    return jsonify(results)

# ====================== HISTORY ====================
@app.route("/history", methods=["GET"])
def history():
    docs = db.collection("sentiments") \
        .order_by("timestamp", direction=firestore.Query.DESCENDING) \
        .limit(200) \
        .stream()

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

# ====================== STATS ====================
@app.route("/stats", methods=["GET"])
def stats():
    docs = db.collection("sentiments").stream()

    total = positive = negative = 0

    for doc in docs:
        s = doc.to_dict().get("sentiment", "")
        total += 1
        if s == "Positive":
            positive += 1
        else:
            negative += 1

    return jsonify({
        "total": total,
        "positive": positive,
        "negative": negative
    })

# ====================== ROOT ====================
@app.route("/")
def home():
    return "🔥 Moodify Backend (DistilBERT + Firebase) Running!"

# ====================== RUN ====================
if __name__ == "__main__":
    app.run(port=5000, debug=True)
