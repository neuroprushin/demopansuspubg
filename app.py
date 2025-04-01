from flask import Flask, render_template_string, request, jsonify, Response, send_file
import requests
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import math
import os

app = Flask(__name__)

# Замени на свои ключи Face++
API_KEY = "v00GHB3kc6VmuZ2Sufqbx0u_qqt3u07I"
API_SECRET = "8H7B985VomLOUazkyPqvD5-KkKW-6D_d"

# Описания черт (добавляем эмоции и советы)
DESCRIPTIONS = {
    "Широкая челюсть": "Ты — настоящий танк! Высокая стрессоустойчивость делает тебя непробиваемым в конфликтах. Решаешь задачи без спешки, как стратег. Используй это: бери на себя лидерство в сложных ситуациях!",
    "Узкая челюсть": "Ты — молния! Реагируешь быстро, но стресс может выбить из колеи. Твоя сила — в скорости решений, но будь осторожен с импульсивностью. Совет: дыши глубже в хаосе, и ты всех порвёшь!",
    "Средняя челюсть": "Ты — баланс! Умеренная устойчивость к стрессу и гибкость в конфликтах — твои козыри. Иногда можешь сорваться, но это твой драйв. Двигайся дальше: найди золотую середину и веди команду!"
}

# Пороги (заглушки, откалибруем)
THRESHOLDS = {
    "челюсть": {
        "широкая": 1.1,  # jaw_ratio > 1.1
        "средняя_мин": 0.9,
        "средняя_макс": 1.1,
        "узкая": 0.9  # jaw_ratio < 0.9
    }
}

def resize_image(image_data, max_size=800):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def analyze_face_with_facepp(image_data):
    url = "https://api-us.faceplusplus.com/facepp/v3/detect"
    files = {"image_file": image_data}
    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_landmark": 2,
        "return_attributes": "headpose"
    }
    response = requests.post(url, files=files, data=data)
    if response.status_code == 413:
        print(f"Face++ Error: 413 Request Entity Too Large")
        return {"error": "Изображение слишком большое"}
    if response.status_code != 200:
        print(f"Face++ Error: {response.status_code}, {response.text}")
        return {"error": f"Ошибка Face++: {response.text}"}
    print("JSON от Face++:", json.dumps(response.json(), ensure_ascii=False))
    return response.json()

def get_landmarks(data):
    if "faces" not in data or not data["faces"]:
        print("Нет лиц в данных Face++")
        return {}
    return data["faces"][0]["landmark"]

def get_coords(point):
    if not point or "x" not in point or "y" not in point:
        print(f"Отсутствуют координаты в {point}")
        return 0, 0
    return point["x"], point["y"]

def distance(point1, point2):
    x1, y1 = get_coords(point1)
    x2, y2 = get_coords(point2)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0 else 0

def check_asymmetry(nose_tip, jaw_left, jaw_right, face_width):
    nose_x, nose_y = get_coords(nose_tip)
    left_x, left_y = get_coords(jaw_left)
    right_x, right_y = get_coords(jaw_right)
    left_dist = ((left_x - nose_x) ** 2 + (left_y - nose_y) ** 2) ** 0.5
    right_dist = ((right_x - nose_x) ** 2 + (right_y - nose_y) ** 2) ** 0.5
    asymmetry = abs(left_dist - right_dist) / face_width if face_width > 0 else 0
    return asymmetry, left_dist, right_dist

def analyze_landmarks(landmarks, headpose, img, show_details=False):
    if not landmarks:
        cv2.putText(img, "Нет лица", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return {}, "Нет лица", img, {}

    pitch, yaw, roll = headpose.get("pitch_angle", 0), headpose.get("yaw_angle", 0), headpose.get("roll_angle", 0)
    if abs(pitch) > 10 or abs(yaw) > 10 or abs(roll) > 5:
        cv2.putText(img, "Фото не в анфас", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return {}, "Фото не в анфас", img, {}

    result_texts = []
    height, width = img.shape[:2]

    # Точки для измерения
    temple_left = landmarks.get("contour_left1")
    temple_right = landmarks.get("contour_right1")
    jaw_left = landmarks.get("contour_left9")
    jaw_right = landmarks.get("contour_right9")
    nose_tip = landmarks.get("nose_tip")

    # Рисуем точки
    for point in [temple_left, temple_right, jaw_left, jaw_right, nose_tip]:
        x, y = get_coords(point)
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), 7, (255, 0, 0), 3)

    # Измеряем расстояния
    face_width = distance(temple_left, temple_right)
    jaw_width = distance(jaw_left, jaw_right)
    jaw_ratio = jaw_width / face_width if face_width > 0 else 0

    # Проверка асимметрии
    asymmetry, left_dist, right_dist = check_asymmetry(nose_tip, jaw_left, jaw_right, face_width)
    asymmetry_warning = asymmetry > 0.1  # Если асимметрия > 10% от face_width

    # Рисуем линии между точками (визуализация расстояний)
    x_tl, y_tl = get_coords(temple_left)
    x_tr, y_tr = get_coords(temple_right)
    x_jl, y_jl = get_coords(jaw_left)
    x_jr, y_jr = get_coords(jaw_right)
    cv2.line(img, (int(x_tl), int(y_tl)), (int(x_tr), int(y_tr)), (0, 255, 0), 4)  # Линия между висками
    cv2.line(img, (int(x_jl), int(y_jl)), (int(x_jr), int(y_jr)), (0, 255, 0), 4)  # Линия между краями челюсти

    # Классификация челюсти
    if jaw_ratio > THRESHOLDS["челюсть"]["широкая"]:
        jaw_trait = "Широкая челюсть"
    elif jaw_ratio < THRESHOLDS["челюсть"]["узкая"]:
        jaw_trait = "Узкая челюсть"
    else:
        jaw_trait = "Средняя челюсть"

    # Выводим информацию на изображение
    if show_details:
        cv2.putText(img, f"Face Width: {face_width:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Jaw Width: {jaw_width:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Jaw Ratio: {jaw_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Asymmetry: {asymmetry:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if asymmetry_warning:
            cv2.putText(img, "Warning: Face asymmetry detected", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if jaw_trait in DESCRIPTIONS:
        result_texts.append(DESCRIPTIONS[jaw_trait])
        cv2.putText(img, jaw_trait, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    result_text = " ".join(result_texts) if result_texts else "Не удалось определить"
    _, buffer = cv2.imencode('.jpg', img)
    img_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    log_data = {
        "face_width": face_width,
        "jaw_width": jaw_width,
        "jaw_ratio": jaw_ratio,
        "asymmetry": asymmetry,
        "left_jaw_dist": left_dist,
        "right_jaw_dist": right_dist,
        "headpose": headpose,
        "temple_left": get_coords(temple_left),
        "temple_right": get_coords(temple_right),
        "jaw_left": get_coords(jaw_left),
        "jaw_right": get_coords(jaw_right),
        "nose_tip": get_coords(nose_tip)
    }

    with open('logs.json', 'a') as f:
        json.dump(log_data, f, ensure_ascii=False)
        f.write('\n')

    return {}, result_text, img_str, log_data

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Калибровка челюсти</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #1a1a1a; color: #fff; text-align: center; padding: 50px; }
        h1 { color: #00ff00; }
        #prototype { position: absolute; top: 10px; right: 10px; color: #ff0000; font-size: 14px; }
        button { background-color: #00ff00; color: #000; padding: 10px 20px; border: none; cursor: pointer; font-size: 16px; margin: 5px; }
        button:hover { background-color: #00cc00; }
        #result { margin-top: 20px; font-size: 18px; }
        #imageResult { margin-top: 20px; max-width: 100%; }
        #downloadBtn { display: none; margin-top: 10px; }
    </style>
</head>
<body>
    <div id="prototype">Beta-porogi-0.2</div>
    <h1>Калибровка челюсти</h1>
    <input type="file" id="photoInput" accept="image/*">
    <button onclick="analyzeFace()">Анализировать</button>
    <button id="showDetailsBtn" onclick="toggleDetails()">Показать детали</button>
    <button id="downloadBtn" onclick="downloadLogs()">Скачать логи</button>
    <p id="result"></p>
    <img id="imageResult" style="display: none;">
    <script>
        let showDetails = false;
        async function analyzeFace() {
            const file = document.getElementById('photoInput').files[0];
            if (!file) { document.getElementById('result').innerText = "Загрузи фото"; return; }
            const formData = new FormData(); formData.append('photo', file);
            try {
                const response = await fetch('/analyze?details=' + showDetails, { method: 'POST', body: formData });
                const data = await response.json();
                document.getElementById('result').innerText = data.result || "Ошибка";
                if (data.image) {
                    document.getElementById('imageResult').src = data.image;
                    document.getElementById('imageResult').style.display = 'block';
                    document.getElementById('downloadBtn').style.display = 'block';
                } else document.getElementById('imageResult').style.display = 'none';
            } catch (error) { document.getElementById('result').innerText = "Ошибка сети"; }
        }
        function toggleDetails() { showDetails = !showDetails; document.getElementById('showDetailsBtn').innerText = showDetails ? "Скрыть детали" : "Показать детали"; if (document.getElementById('photoInput').files[0]) analyzeFace(); }
        function downloadLogs() {
            window.location.href = '/download_logs';
        }
    </script>
</body>
</html>
''')

@app.route('/analyze', methods=['POST'])
def analyze():
    show_details = request.args.get('details', 'false').lower() == 'true'
    file = request.files['photo']
    if not file:
        return jsonify({"result": "Нет фото"})

    image_data = file.read()
    image_data = resize_image(image_data, max_size=800)

    face_data = analyze_face_with_facepp(image_data)
    if "error" in face_data:
        return jsonify({"result": face_data["error"]})

    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    landmarks = get_landmarks(face_data)
    headpose = face_data["faces"][0]["attributes"]["headpose"] if "faces" in face_data and face_data["faces"] else {}
    traits, result_text, image_base64, log_data = analyze_landmarks(landmarks, headpose, img, show_details)

    return jsonify({"result": result_text, "image": image_base64})

@app.route('/download_logs')
def download_logs():
    try:
        return send_file('logs.json', as_attachment=True, download_name='logs.json')
    except FileNotFoundError:
        return "Лог-файл не найден. Обработайте хотя бы одно фото.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)