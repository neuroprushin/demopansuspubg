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
import traceback

app = Flask(__name__)

API_KEY = "v00GHB3kc6VmuZ2Sufqbx0u_qqt3u07I"
API_SECRET = "8H7B985VomLOUazkyPqvD5-KkKW-6D_d"

DESCRIPTIONS = {
    "Широкая челюсть": "Ты — настоящий танк! Высокая стрессоустойчивость делает тебя непробиваемым в конфликтах. Решаешь задачи без спешки, как стратег. Используй это: бери на себя лидерство в сложных ситуациях!",
    "Узкая челюсть": "Ты — молния! Реагируешь быстро, но стресс может выбить из колеи. Твоя сила — в скорости решений, но будь осторожен с импульсивностью. Совет: дыши глубже в хаосе, и ты всех порвёшь!",
    "Средняя челюсть": "Ты — баланс! Умеренная устойчивость к стрессу и гибкость в конфликтах — твои козыри. Иногда можешь сорваться, но это твой драйв. Двигайся дальше: найди золотую середину и веди команду!"
}

THRESHOLDS = {
    "челюсть": {
        "широкая": -35,  # jaw_diff > -35
        "средняя_мин": -45,
        "средняя_макс": -35,
        "узкая": -45  # jaw_diff <= -45
    }
}

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

def analyze_face_with_facepp(image_data):
    url = "https://api-us.faceplusplus.com/facepp/v3/detect"
    files = {"image_file": image_data}
    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_landmark": 2,
        "return_attributes": "headpose"
    }
    try:
        response = requests.post(url, files=files, data=data, timeout=10)
        if response.status_code == 413:
            print(f"Face++ Error: 413 Request Entity Too Large")
            return {"error": "Изображение слишком большое"}
        if response.status_code != 200:
            print(f"Face++ Error: {response.status_code}, {response.text}")
            return {"error": f"Ошибка Face++: {response.status_code} - {response.text}"}
        print("JSON от Face++:", json.dumps(response.json(), ensure_ascii=False))
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Face++ Request Failed: {str(e)}")
        return {"error": f"Ошибка связи с Face++: {str(e)}"}

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
    # Проверка headpose в самом начале
    pitch, yaw, roll = headpose.get("pitch_angle", 0), headpose.get("yaw_angle", 0), headpose.get("roll_angle", 0)
    if abs(pitch) > 20 or abs(yaw) > 20 or abs(roll) > 5:
        return {}, "Фото не в анфас", None, {}

    if not landmarks:
        return {}, "Нет лица", None, {}

    result_texts = []
    height, width = img.shape[:2]

    temple_left = landmarks.get("contour_left1")
    temple_right = landmarks.get("contour_right1")
    jaw_left = landmarks.get("contour_left9")
    jaw_right = landmarks.get("contour_right9") 
    nose_tip = landmarks.get("nose_tip")

    for point in [temple_left, temple_right, jaw_left, jaw_right, nose_tip]:
        x, y = get_coords(point)
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), 7, (255, 0, 0), 3)

    face_width = distance(temple_left, temple_right)
    jaw_width = distance(jaw_left, jaw_right)
    jaw_ratio = jaw_width / face_width if face_width > 0 else 0
    jaw_diff = jaw_width - face_width

    asymmetry, left_dist, right_dist = check_asymmetry(nose_tip, jaw_left, jaw_right, face_width)
    asymmetry_warning = asymmetry > 0.1
    headpose_warning = abs(pitch) > 8 or abs(yaw) > 8

    if asymmetry_warning or headpose_warning:
        warning_text = "Предупреждение: Высокая асимметрия или наклон головы"
        cv2.putText(img, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    x_tl, y_tl = get_coords(temple_left)
    x_tr, y_tr = get_coords(temple_right)
    x_jl, y_jl = get_coords(jaw_left)
    x_jr, y_jr = get_coords(jaw_right)
    cv2.line(img, (int(x_tl), int(y_tl)), (int(x_tr), int(y_tr)), (0, 255, 0), 4)
    cv2.line(img, (int(x_jl), int(y_jl)), (int(x_jr), int(y_jr)), (0, 255, 0), 4)

    if jaw_diff > THRESHOLDS["челюсть"]["широкая"]:
        jaw_trait = "Широкая челюсть"
    elif jaw_diff < THRESHOLDS["челюсть"]["узкая"]:
        jaw_trait = "Узкая челюсть"
    else:
        jaw_trait = "Средняя челюсть"

    if show_details:
        cv2.putText(img, f"Face Width: {face_width:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Jaw Width: {jaw_width:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Jaw Ratio: {jaw_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Jaw Diff: {jaw_diff:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Asymmetry: {asymmetry:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if asymmetry_warning:
            cv2.putText(img, "Warning: Face asymmetry detected", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if headpose_warning:
            cv2.putText(img, "Warning: Head pose may affect results", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if jaw_trait in DESCRIPTIONS:
        result_texts.append(DESCRIPTIONS[jaw_trait])
        cv2.putText(img, jaw_trait, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    result_text = " ".join(result_texts) if result_texts else "Не удалось определить"

    # Кодирование изображения с проверками
    try:
        retval, buffer = cv2.imencode('.jpg', img)
        if not retval:
            raise ValueError("Не удалось закодировать изображение")
        if not isinstance(buffer, np.ndarray):
            raise ValueError(f"buffer не является ndarray, тип: {type(buffer)}")
        if buffer.size == 0:
            raise ValueError("buffer пустой")
        if buffer.dtype != np.uint8:
            raise ValueError(f"buffer имеет неправильный тип данных: {buffer.dtype}, ожидается uint8")
        print(f"Тип buffer: {type(buffer)}, размер: {buffer.size}, dtype: {buffer.dtype}")
        # Преобразуем buffer в bytes
        buffer_bytes = buffer.tobytes()
        image_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer_bytes).decode('utf-8')
        if not isinstance(image_base64, str):
            raise ValueError(f"image_base64 не строка, тип: {type(image_base64)}")
        print(f"Тип image_base64: {type(image_base64)}")
    except Exception as e:
        print(f"Ошибка при кодировании изображения: {str(e)}")
        return {}, "Ошибка: Не удалось закодировать изображение", None, {}

    log_data = {
        "face_width": face_width,
        "jaw_width": jaw_width,
        "jaw_ratio": jaw_ratio,
        "jaw_diff": jaw_diff,
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

    return {}, result_text, image_base64, log_data

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
    <div id="prototype">Beta-porogi-0.9</div>
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
                await new Promise(resolve => setTimeout(resolve, 1000));
                const response = await fetch('/analyze?details=' + showDetails, { method: 'POST', body: formData });
                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`Сервер вернул ошибку: ${response.status} - ${text}`);
                }
                const data = await response.json();
                document.getElementById('result').innerText = data.result || "Ошибка";
                if (data.image) {
                    document.getElementById('imageResult').src = data.image;
                    document.getElementById('imageResult').style.display = 'block';
                    document.getElementById('downloadBtn').style.display = 'block';
                } else {
                    document.getElementById('imageResult').style.display = 'none';
                    document.getElementById('downloadBtn').style.display = 'none';
                }
            } catch (error) {
                document.getElementById('result').innerText = "Ошибка: " + error.message;
            }
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
    try:
        show_details = request.args.get('details', 'false').lower() == 'true'
        if 'photo' not in request.files:
            return jsonify({"result": "Нет фото"}), 400

        file = request.files['photo']
        image_data = file.read()
        if not image_data:
            return jsonify({"result": "Файл пустой"}), 400

        print("Размер image_data:", len(image_data))
        image_data = resize_image(image_data, max_size=800)
        print("Размер image_data после resize:", len(image_data))

        face_data = analyze_face_with_facepp(image_data)
        if "error" in face_data:
            return jsonify({"result": face_data["error"]}), 400

        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"result": "Не удалось декодировать изображение"}), 400

        # Проверка img
        if not isinstance(img, np.ndarray):
            return jsonify({"result": "Некорректное изображение: img не является ndarray"}), 400
        if len(img.shape) != 3 or img.shape[2] != 3:
            return jsonify({"result": f"Некорректное изображение: неправильные размеры {img.shape}, ожидается (height, width, 3)"}), 400
        if img.dtype != np.uint8:
            return jsonify({"result": f"Некорректное изображение: неправильный тип данных {img.dtype}, ожидается uint8"}), 400
        print(f"img shape: {img.shape}, dtype: {img.dtype}")

        landmarks = get_landmarks(face_data)
        headpose = face_data["faces"][0]["attributes"]["headpose"] if "faces" in face_data and face_data["faces"] else {}
        traits, result_text, image_base64, log_data = analyze_landmarks(landmarks, headpose, img, show_details)

        if image_base64 is None:
            return jsonify({"result": result_text}), 400

        return jsonify({"result": result_text, "image": image_base64})
    except Exception as e:
        print(f"Ошибка в /analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"result": f"Внутренняя ошибка сервера: {str(e)}"}), 500

@app.route('/download_logs')
def download_logs():
    try:
        return send_file('logs.json', as_attachment=True, download_name='logs.json')
    except FileNotFoundError:
        return jsonify({"result": "Лог-файл не найден. Обработайте хотя бы одно фото."}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)