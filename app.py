import os
import cv2
import numpy as np
import json
import base64
import requests
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO

app = Flask(__name__)

# Face++ API credentials
API_KEY = "v00GHB3kc6VmuZ2Sufqbx0u_qqt3u07I"
API_SECRET = "8H7B985VomLOUazkyPqvD5-KkKW-6D_d"
FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Thresholds for jaw classification (Beta-porogi-1.3)
THRESHOLDS = {
    "челюсть": {
        "узкая": 0.8077,  # jaw_ratio <= 0.8077
        "узко_средняя_макс": 0.8600,
        "средняя_мин": 0.8600,
        "средняя_макс": 0.8800,
        "средне_широкая_макс": 0.8968,
        "широкая": 0.8968  # jaw_ratio >= 0.8968
    }
}

# Function to resize image
def resize_image(image_data, max_size=800):
    try:
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось декодировать изображение")
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    except Exception as e:
        print(f"Ошибка в resize_image: {str(e)}")
        raise

# Function to get coordinates from landmark point
def get_coords(point):
    return point.get("x", 0), point.get("y", 0)

# Function to calculate distance between two points
def distance(point1, point2):
    x1, y1 = get_coords(point1)
    x2, y2 = get_coords(point2)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0 else 0

# Function to analyze landmarks
def analyze_landmarks(landmarks):
    try:
        temple_left = landmarks.get("contour_left1")
        temple_right = landmarks.get("contour_right1")
        jaw_left = landmarks.get("contour_left9")
        jaw_right = landmarks.get("contour_right9")
        nose_tip = landmarks.get("nose_tip")

        if not all([temple_left, temple_right, jaw_left, jaw_right, nose_tip]):
            raise ValueError("Не удалось извлечь все ключевые точки")

        face_width = distance(temple_left, temple_right)
        jaw_width = distance(jaw_left, jaw_right)
        jaw_ratio = jaw_width / face_width if face_width > 0 else 0
        jaw_diff = jaw_width - face_width

        left_jaw_dist = distance(jaw_left, nose_tip)
        right_jaw_dist = distance(jaw_right, nose_tip)
        asymmetry = abs(left_jaw_dist - right_jaw_dist) / max(left_jaw_dist, right_jaw_dist) if max(left_jaw_dist, right_jaw_dist) > 0 else 0

        return {
            "face_width": face_width,
            "jaw_width": jaw_width,
            "jaw_ratio": jaw_ratio,
            "jaw_diff": jaw_diff,
            "asymmetry": asymmetry,
            "left_jaw_dist": left_jaw_dist,
            "right_jaw_dist": right_jaw_dist,
            "temple_left": [temple_left["x"], temple_left["y"]],
            "temple_right": [temple_right["x"], temple_right["y"]],
            "jaw_left": [jaw_left["x"], jaw_left["y"]],
            "jaw_right": [jaw_right["x"], jaw_right["y"]],
            "nose_tip": [nose_tip["x"], nose_tip["y"]]
        }
    except Exception as e:
        print(f"Ошибка в analyze_landmarks: {str(e)}")
        raise

# Function to call Face++ API
def call_facepp_api(image_data):
    try:
        files = {"image_file": image_data}
        params = {
            "api_key": API_KEY,
            "api_secret": API_SECRET,
            "return_landmark": "all",
            "return_attributes": "headpose"
        }
        response = requests.post(FACEPP_URL, files=files, data=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("faces"):
            raise ValueError("Лицо не обнаружено на изображении")

        face = data["faces"][0]
        landmarks = face["landmark"]
        headpose = face["attributes"]["headpose"]

        return landmarks, headpose
    except Exception as e:
        print(f"Ошибка в call_facepp_api: {str(e)}")
        raise

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to process image
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Изображение не найдено"}), 400

        image_file = request.files['image']
        image_data = image_file.read()

        # Resize image
        resized_image = resize_image(image_data)

        # Call Face++ API
        landmarks, headpose = call_facepp_api(resized_image)

        # Analyze landmarks
        analysis = analyze_landmarks(landmarks)
        analysis["headpose"] = headpose

        # Classify jaw trait
        jaw_ratio = analysis["jaw_ratio"]
        if jaw_ratio <= THRESHOLDS["челюсть"]["узкая"]:
            jaw_trait = "Узкая челюсть"
        elif jaw_ratio <= THRESHOLDS["челюсть"]["узко_средняя_макс"]:
            jaw_trait = "Узко-средняя челюсть"
        elif jaw_ratio <= THRESHOLDS["челюсть"]["средняя_макс"]:
            jaw_trait = "Средняя челюсть"
        elif jaw_ratio < THRESHOLDS["челюсть"]["средне_широкая_макс"]:
            jaw_trait = "Средне-широкая челюсть"
        else:
            jaw_trait = "Широкая челюсть"

        analysis["jaw_trait"] = jaw_trait

        # Save to logs
        log_entry = analysis.copy()
        log_file = "logs.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to download logs
@app.route('/download_logs')
def download_logs():
    log_file = "logs.json"
    if not os.path.exists(log_file):
        return jsonify({"error": "Логи не найдены"}), 404
    return send_file(log_file, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))