import os
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext

MODEL_PATH = "genre_fasttext_model_small.ftz"
MODEL_URL = "https://cvsqzmoyqfvhwdxtvqhb.supabase.co/storage/v1/object/public/models/genre_fasttext_model_small.ftz"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading FastText model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Confirm the file is valid before loading
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    raise ValueError("⚠️ Model download failed or incomplete.")

print("✅ Model ready, loading into memory...")
model = fasttext.load_model(MODEL_PATH)

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
