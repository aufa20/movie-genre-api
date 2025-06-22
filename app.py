from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext
import os

# Load trained FastText model
model = fasttext.load_model("genre_fasttext_model.ftz")

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
