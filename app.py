from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os  # Required for os.environ

# Load model
with open("genre_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    desc = data.get('description', '')
    pred = model.predict([desc])[0]
    return jsonify({'genre': pred})

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )