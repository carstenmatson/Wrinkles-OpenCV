# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

# ✅ Import the wrinkles detection module (since it's in the root directory)
from wrinkles import detect_wrinkles

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ✅ Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def normalize_score(score, min_value=25, max_value=100):
    """Ensure wrinkle scores stay between 25 and 100."""
    return int(max(min(score, max_value), min_value))

@app.route('/')
def home():
    """Base route."""
    return jsonify({"message": "✅ Wrinkle detection API is live!"}), 200

@app.route('/healthz')
def health():
    """Health check."""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze wrinkles from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    skin_tone = int(request.form.get('skin_tone', 5))  # Default to medium skin tone (5)
    filename = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filename)

    # ✅ Directly compute wrinkle score for the whole image
    image = cv2.imread(filename)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    raw_score = detect_wrinkles(image, skin_tone)
    wrinkle_score = normalize_score(raw_score)

    return jsonify({"wrinkle_score": wrinkle_score})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)


