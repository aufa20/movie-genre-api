import os
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext

MODEL_PATH = "genre_fasttext_model_small.ftz"
MODEL_URL = "https://cvsqzmoyqfvhwdxtvqhb.supabase.co/storage/v1/object/public/models/genre_fasttext_model_small.ftz"

try:
    if not os.path.exists(MODEL_PATH):
        print("📥 Attempting to download model from Supabase...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print("❌ File does not exist after download.")
        raise ValueError("⚠️ Model download failed.")

    size = os.path.getsize(MODEL_PATH)
    print(f"📦 File downloaded. Size: {size / 1024 / 1024:.2f} MB")

    if size < 1_000_000:
        raise ValueError("⚠️ Model download incomplete or corrupted.")

    model = fasttext.load_model(MODEL_PATH)
    print("✅ FastText model loaded successfully.")

except Exception as e:
    print(f"❌ Exception during model loading: {e}")
    raise


# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    desc = data.get("description", "").strip()

    if not desc:
        return jsonify({"error": "No description provided"}), 400

    label, confidence = model.predict(desc)
    genre = label[0].replace("__label__", "")
    return jsonify({
        "genre": genre,
        "confidence": round(confidence[0], 4)
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
