import os
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext

MODEL_PATH = "genre_fasttext_model_tiny.ftz"
MODEL_URL = "https://www.dropbox.com/scl/fi/jg86kufq6nh8vj2n3ujhe/genre_fasttext_model_tiny.ftz?rlkey=syj7tc35l8j42mlj4sk4xircp&dl=1"

# Download model if missing
try:
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Dropbox...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    size = os.path.getsize(MODEL_PATH)
    print(f"📦 Model file size: {size / 1024 / 1024:.2f} MB")
    if size < 1000000:
        raise ValueError("⚠️ Model file is too small or incomplete.")

    model = fasttext.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

# Flask app setup
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
