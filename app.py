from flask import Flask, render_template_string, request, jsonify, Response
import requests
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import math

app = Flask(__name__)

# Замени на свои ключи Face++
API_KEY = "v00GHB3kc6VmuZ2Sufqbx0u_qqt3u07I"
API_SECRET = "8H7B985VomLOUazkyPqvD5-KkKW-6D_d"

# Описания черт (только челюсть)
DESCRIPTIONS = {
    "Широкая челюсть": "Ты высоко стрессоустойчив, спокойно относишься к конфликтам и предпочитаешь решать задачи постепенно, не торопясь.",
    "Узкая челюсть": "Ты менее устойчив к стрессу, склонен к резким реакциям и стремишься быстро устранить проблемы, иногда импульсивно.",
    "Средняя челюсть": "Ты умеренно устойчив к стрессу, можешь быть гибким в конфликтах, но иногда реагируешь импульсивно."
}

# Сырые пороги (будем калибровать)
THRESHOLDS = {
    "челюсть": {
        "широкая": 0.15,  # Заглушка, откалибруем
        "средняя_мин": 0.05,
        "средняя_макс": 0.15,
        "узкая": 0.05
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

def draw_tilted_temple_lines(img, temple_left, temple_right, headpose, height, width, jaw_left, jaw_right):
    x_tl, y_tl = get_coords(temple_left)
    x_tr, y_tr = get_coords(temple_right)
    x_jl, y_jl = get_coords(jaw_left)
    x_jr, y_jr = get_coords(jaw_right)
    
    roll = headpose.get("roll_angle", 0)
    pitch = headpose.get("pitch_angle", 0)
    yaw = headpose.get("yaw_angle", 0)
    
    angle = math.radians(roll) + math.radians(pitch) * 0.5 + math.radians(yaw) * 0.1
    
    length_left = abs(y_jl - y_tl)
    length_right = abs(y_jr - y_tr)
    
    end_x_tl = x_tl + length_left * math.sin(angle)
    end_y_tl = y_tl + length_left * math.cos(angle)
    end_x_tr = x_tr + length_right * math.sin(angle)
    end_y_tr = y_tr + length_right * math.cos(angle)
    
    # Корректировка для равенства расстояний
    dist_left = abs(x_jl - end_x_tl) if end_x_tl != x_tl else abs(y_jl - end_y_tl)
    dist_right = abs(x_jr - end_x_tr) if end_x_tr != x_tr else abs(y_jr - end_y_tr)
    tolerance = 0.5
    for _ in range(10):
        if abs(dist_left - dist_right) > tolerance:
            angle_correction = (dist_left - dist_right) * 0.01
            angle += math.radians(angle_correction)
            end_x_tl = x_tl + length_left * math.sin(angle)
            end_y_tl = y_tl + length_left * math.cos(angle)
            end_x_tr = x_tr + length_right * math.sin(angle)
            end_y_tr = y_tr + length_right * math.cos(angle)
            dist_left = abs(x_jl - end_x_tl) if end_x_tl != x_tl else abs(y_jl - end_y_tl)
            dist_right = abs(x_jr - end_x_tr) if end_x_tr != x_tr else abs(y_jr - end_y_tr)
        else:
            break
    
    end_x_tl = max(0, min(width - 1, int(end_x_tl)))
    end_y_tl = max(0, min(height - 1, int(end_y_tl)))
    end_x_tr = max(0, min(width - 1, int(end_x_tr)))
    end_y_tr = max(0, min(height - 1, int(end_y_tr)))
    
    cv2.line(img, (int(x_tl), int(y_tl)), (end_x_tl, end_y_tl), (0, 255, 0), 4)
    cv2.line(img, (int(x_tr), int(y_tr)), (end_x_tr, end_y_tr), (0, 255, 0), 4)
    
    return img, (end_x_tl, end_y_tl), (end_x_tr, end_y_tr), dist_left, dist_right

def check_jaw_position(jaw_left, jaw_right, temple_left, temple_right, end_temple_left, end_temple_right):
    x_jl, y_jl = get_coords(jaw_left)
    x_jr, y_jr = get_coords(jaw_right)
    x_tl, y_tl = get_coords(temple_left)
    x_tr, y_tr = get_coords(temple_right)
    if 0 in [x_jl, x_jr, x_tl, x_tr]:
        return "внутри_линий"
    if (x_jl < min(x_tl, end_temple_left[0]) and x_jr > max(x_tr, end_temple_right[0])) or \
       (x_jl > max(x_tl, end_temple_left[0]) and x_jr < min(x_tr, end_temple_right[0])):
        return "за_линиями"
    return "внутри_линий"

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
    for point_name, point in landmarks.items():
        x, y = get_coords(point)
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), 3, (255, 255, 255), 3)

    face_left = landmarks.get("contour_left1")
    face_right = landmarks.get("contour_right1")
    jaw_left = landmarks.get("contour_left9")
    jaw_right = landmarks.get("contour_right9")
    temple_left = landmarks.get("contour_left1")
    temple_right = landmarks.get("contour_right1")

    for point in [face_left, face_right, jaw_left, jaw_right, temple_left, temple_right]:
        x, y = get_coords(point)
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), 7, (255, 0, 0), 3)

    img, end_temple_left, end_temple_right, dist_left, dist_right = draw_tilted_temple_lines(img, temple_left, temple_right, headpose, height, width, jaw_left, jaw_right)

    jaw_pos = check_jaw_position(jaw_left, jaw_right, temple_left, temple_right, end_temple_left, end_temple_right)
    face_width = distance(face_left, face_right)
    avg_dist = (dist_left + dist_right) / 2 if (dist_left > 0 and dist_right > 0) else 0
    normalized_dist = avg_dist / face_width if face_width > 0 else 0
    dist_diff = abs(dist_left - dist_right)

    if show_details:
        cv2.putText(img, f"Dist Left: {dist_left:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Dist Right: {dist_right:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Avg Dist: {avg_dist:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Norm Dist: {normalized_dist:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Diff: {dist_diff:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Inside Lines: {jaw_pos == 'внутри_линий'}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if jaw_pos == "за_линиями":
        if normalized_dist > THRESHOLDS["челюсть"]["широкая"]:
            jaw_trait = "Широкая челюсть"
        else:
            jaw_trait = "Средняя челюсть"
    else:
        if normalized_dist < THRESHOLDS["челюсть"]["узкая"]:
            jaw_trait = "Узкая челюсть"
        else:
            jaw_trait = "Средняя челюсть"

    if jaw_trait in DESCRIPTIONS:
        result_texts.append(DESCRIPTIONS[jaw_trait])
        cv2.putText(img, jaw_trait, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    result_text = " ".join(result_texts) if result_texts else "Не удалось определить"
    _, buffer = cv2.imencode('.jpg', img)
    img_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    log_data = {
        "dist_left": dist_left,
        "dist_right": dist_right,
        "avg_dist": avg_dist,
        "normalized_dist": normalized_dist,
        "dist_diff": dist_diff,
        "inside_lines": jaw_pos == "внутри_линий",
        "headpose": headpose,
        "face_width": face_width,
        "jaw_left": get_coords(jaw_left),
        "jaw_right": get_coords(jaw_right),
        "temple_left": get_coords(temple_left),
        "temple_right": get_coords(temple_right),
        "end_temple_left": end_temple_left,
        "end_temple_right": end_temple_right
    }

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
    </style>
</head>
<body>
    <div id="prototype">Beta-porogi-0.1</div>
    <h1>Калибровка челюсти</h1>
    <input type="file" id="photoInput" accept="image/*">
    <button onclick="analyzeFace()">Анализировать</button>
    <button id="showDetailsBtn" onclick="toggleDetails()">Показать детали</button>
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
                } else document.getElementById('imageResult').style.display = 'none';
            } catch (error) { document.getElementById('result').innerText = "Ошибка сети"; }
        }
        function toggleDetails() { showDetails = !showDetails; document.getElementById('showDetailsBtn').innerText = showDetails ? "Скрыть детали" : "Показать детали"; if (document.getElementById('photoInput').files[0]) analyzeFace(); }
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

    with open('logs.json', 'a') as f:
        json.dump(log_data, f, ensure_ascii=False)
        f.write('\n')

    return jsonify({"result": result_text, "image": image_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)