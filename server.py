# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

# ✅ Import modules
from preprocessing.crop_face import detect_and_crop_faces
from preprocessing.extract_regions import detect_and_extract_regions
from main.wrinkles import detect_wrinkles

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ✅ Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_faces")
REGIONS_DIR = os.path.join(BASE_DIR, "data/extracted_regions")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REGIONS_DIR, exist_ok=True)

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

    # ✅ Step 1: Detect and crop face
    processed_path = os.path.join(PROCESSED_DIR, file.filename)
    detect_and_crop_faces(filename, PROCESSED_DIR)

    # ✅ Step 2: Extract facial regions
    detect_and_extract_regions(processed_path, REGIONS_DIR)

    # ✅ Step 3: Compute wrinkle score for each region
    region_scores = {}
    for region in ["forehead", "left_cheek", "right_cheek", "chin"]:
        region_path = os.path.join(REGIONS_DIR, f"{os.path.splitext(file.filename)[0]}_{region}.jpg")
        if os.path.exists(region_path):
            region_image = cv2.imread(region_path)
            raw_score = detect_wrinkles(region_image, skin_tone)
            region_scores[region] = normalize_score(raw_score)

    if not region_scores:
        return jsonify({"error": "No facial regions detected"}), 400

    return jsonify({"wrinkle_scores": region_scores})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

