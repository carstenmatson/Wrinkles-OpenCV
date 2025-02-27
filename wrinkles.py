# wrinkles.py
import cv2
import numpy as np
import mediapipe as mp

def preprocess_image(image):
    """Normalize brightness for consistent wrinkle detection."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def detect_lips(image):
    """Detect lips and create a mask to exclude them from wrinkle detection."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    height, width, _ = image.shape

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = np.zeros((height, width), dtype=np.uint8)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            lip_points = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
            points = [(int(landmarks.landmark[i].x * width), int(landmarks.landmark[i].y * height)) for i in lip_points]
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

    return mask

def adjust_for_skin_tone(value, skin_tone):
    """Adjust wrinkle detection based on skin tone (1-10 scale)."""
    correction_factor = 1 + ((10 - skin_tone) * 0.05)
    return int(max(25, min(100, 100 - (value * correction_factor))))  # Clamp 25-100

def detect_wrinkles(image, skin_tone):
    """Detect facial wrinkles with edge detection."""
    image = preprocess_image(image)
    lips_mask = detect_lips(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges[lips_mask == 255] = 0  # Remove lips from analysis

    wrinkle_score = 100 - (np.sum(edges) / (image.shape[0] * image.shape[1])) * 100
    return adjust_for_skin_tone(wrinkle_score, skin_tone)
