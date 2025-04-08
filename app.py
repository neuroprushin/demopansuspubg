import os
import cv2
import numpy as np
import json
import mediapipe as mp
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO

app = Flask(__name__)

# Инициализация FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Пороги (Beta-porogi-1.9)
THRESHOLDS = {
    "челюсть": {
        "узкая": 0.8077,
        "узко_средняя_макс": 0.8200,
        "средняя_мин": 0.8260,
        "средняя_макс": 0.8400,
        "средне_широкая_макс": 0.8819,
        "широкая": 0.8819
    }
}

def resize_image(image_data, max_size=800):
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

def get_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def analyze_landmarks(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        raise ValueError("Лицо не обнаружено FaceMesh")

    landmarks = results.multi_face_landmarks[0].landmark
    height, width, _ = img.shape

    # Точки висков и челюсти (примерные индексы)
    temple_left = landmarks[127]  # Левый висок
    temple_right = landmarks[356]  # Правый висок
    jaw_left = landmarks[172]  # Левая точка челюсти
    jaw_right = landmarks[397]  # Правая точка челюсти
    nose_tip = landmarks[1]  # Кончик носа

    # Координаты
    x_temple_left, y_temple_left = get_coords(temple_left, width, height)
    x_temple_right, y_temple_right = get_coords(temple_right, width, height)
    x_jaw_left, y_jaw_left = get_coords(jaw_left, width, height)
    x_jaw_right, y_jaw_right = get_coords(jaw_right, width, height)
    x_nose_tip, y_nose_tip = get_coords(nose_tip, width, height)

    # Расчёты
    face_width = distance(x_temple_left, y_temple_left, x_temple_right, y_temple_right)
    jaw_width = distance(x_jaw_left, y_jaw_left, x_jaw_right, y_jaw_right)
    jaw_ratio = jaw_width / face_width if face_width > 0 else 0
    jaw_diff = jaw_width - face_width

    left_jaw_dist = distance(x_jaw_left, y_jaw_left, x_nose_tip, y_nose_tip)
    right_jaw_dist = distance(x_jaw_right, y_jaw_right, x_nose_tip, y_nose_tip)
    asymmetry = abs(left_jaw_dist - right_jaw_dist) / max(left_jaw_dist, right_jaw_dist) if max(left_jaw_dist, right_jaw_dist) > 0 else 0

    # Сравнение с линией от виска вниз
    left_deviation = x_jaw_left - x_temple_left
    right_deviation = x_jaw_right - x_temple_right

    return {
        "face_width": face_width,
        "jaw_width": jaw_width,
        "jaw_ratio": jaw_ratio,
        "jaw_diff": jaw_diff,
        "asymmetry": asymmetry,
        "left_jaw_dist": left_jaw_dist,
        "right_jaw_dist": right_jaw_dist,
        "temple_left": [x_temple_left, y_temple_left],
        "temple_right": [x_temple_right, y_temple_right],
        "jaw_left": [x_jaw_left, y_jaw_left],
        "jaw_right": [x_jaw_right, y_jaw_right],
        "nose_tip": [x_nose_tip, y_nose_tip],
        "left_deviation": left_deviation,
        "right_deviation": right_deviation
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Изображение не найдено"}), 400

        image_file = request.files['image']
        image_data = image_file.read()

        resized_image = resize_image(image_data)
        analysis = analyze_landmarks(resized_image)

        # Классификация
        jaw_ratio = analysis["jaw_ratio"]
        left_deviation = analysis["left_deviation"]
        right_deviation = analysis["right_deviation"]

        if left_deviation < -10 and right_deviation > 10:
            jaw_trait = "Узкая челюсть"
        else:
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

        log_entry = analysis.copy()
        log_file = "logs.json"
        try:
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        logs = json.load(f)
                except json.JSONDecodeError:
                    print(f"Ошибка: logs.json повреждён, создаём новый")
                    logs = []
            else:
                logs = []
            logs.append(log_entry)
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"Ошибка при сохранении логов: {str(e)}")

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_logs')
def download_logs():
    log_file = "logs.json"
    if not os.path.exists(log_file):
        return jsonify({"error": "Логи не найдены"}), 404
    return send_file(log_file, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))