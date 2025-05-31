from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import logging
import os
import time

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DEBUG_FOLDER = 'debug'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load BISINDO Model
try:
    model = load_model('model/sign_model.h5')
    logger.info("✅ Model loaded successfully")
    input_shape = (50, 50)  # Ukuran input disesuaikan di sini, pastikan sesuai dengan model baru Anda
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    model = None

# BISINDO Labels (diperbarui untuk A-Z)
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# Gesture Mapping
GESTURES = {
    "FIST": "✊",
    "OPEN": "✋",
    "POINT": "☝️",
    "OK": "👌",
    "ROCK": "🤘",
    "UNKNOWN": "❓"
}

def recognize_gesture(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    
    tips = {
        'thumb': landmarks[4],
        'index': landmarks[8],
        'middle': landmarks[12],
        'ring': landmarks[16],
        'pinky': landmarks[20]
    }
    
    thumb_index_dist = np.linalg.norm(np.array(tips['thumb']) - np.array(tips['index']))
    index_middle_dist = np.linalg.norm(np.array(tips['index']) - np.array(tips['middle']))
    
    # Logika pengenalan gesture dasar, bisa disesuaikan atau diperluas
    if tips['thumb'][0] > tips['index'][0] and \
       tips['index'][1] < landmarks[6][1] and \
       tips['middle'][1] < landmarks[10][1] and \
       tips['ring'][1] < landmarks[14][1] and \
       tips['pinky'][1] < landmarks[18][1] and \
       thumb_index_dist > 0.05 : # Jari terbuka
        return "OPEN"
    elif tips['index'][1] < landmarks[5][1] and \
         tips['middle'][1] > landmarks[9][1] and \
         tips['ring'][1] > landmarks[13][1] and \
         tips['pinky'][1] > landmarks[17][1] and \
         np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[8])) > 0.1: # Menunjuk
        return "POINT"
    elif np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[8])) < 0.07 and \
         tips['middle'][1] < landmarks[10][1] and \
         tips['ring'][1] < landmarks[14][1] and \
         tips['pinky'][1] < landmarks[18][1]: # OK
        return "OK"
    # Kondisi FIST mungkin perlu penyesuaian berdasarkan jarak landmark
    elif all(np.linalg.norm(np.array(tips[finger]) - np.array(landmarks[0])) < 0.15 for finger in ['index', 'middle', 'ring', 'pinky']): # Perkiraan FIST
        return "FIST"
    else:
        return "UNKNOWN"

def process_image(img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True, # Diubah menjadi False untuk real-time video, True untuk image tunggal
        max_num_hands=1,
        min_detection_confidence=0.5, # Sedikit diturunkan untuk fleksibilitas
        min_tracking_confidence=0.5) as hands:
        
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, None, "No hand detected"
        
        hand_landmarks = results.multi_hand_landmarks[0]
        # Gesture recognition bisa lebih kompleks, ini contoh dasar
        gesture = recognize_gesture(hand_landmarks) 
        gesture_symbol = GESTURES.get(gesture, GESTURES["UNKNOWN"])

        h, w = img_array.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

        # Expansion Bounding Box agar lebih luas
        expand_ratio = 0.25 # Meningkatkan area crop
        x_padding = (x_max - x_min) * expand_ratio
        y_padding = (y_max - y_min) * expand_ratio

        x_min = max(0, x_min - x_padding)
        y_min = max(0, y_min - y_padding)
        x_max = min(w, x_max + x_padding)
        y_max = min(h, y_max + y_padding)

        hand_crop = img_array[int(y_min):int(y_max), int(x_min):int(x_max)]
        if hand_crop.size == 0:
            return None, gesture_symbol, "Hand cropping failed"
        
        # Resize ke input shape model
        hand_resized = cv2.resize(hand_crop, input_shape) 
        hand_normalized = hand_resized.astype('float32') / 255.0
        hand_ready = np.expand_dims(hand_normalized, axis=0)

        return hand_ready, gesture_symbol, None

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_array is None:
            return jsonify({'error': 'Invalid image data'}), 400

        input_array, gesture, error = process_image(img_array)
        if error:
            # Jika hanya error deteksi tangan, tetap tampilkan gesture jika ada
            if gesture:
                 return jsonify({'error': error, 'gesture': gesture, 'class': '-', 'confidence': 0}), 200 # Atau 400 jika error dianggap fatal
            return jsonify({'error': error}), 400


        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        preds = model.predict(input_array)
        pred_class_idx = int(np.argmax(preds))
        confidence = float(preds[0][pred_class_idx])

        predicted_letter = LABELS.get(pred_class_idx, 'Unknown')
        
        # Log hasil prediksi
        logger.info(f"Predicted: {predicted_letter}, Confidence: {confidence:.2f}, Gesture: {gesture}, Time: {time.time() - start_time:.2f}s")


        return jsonify({
            'class': predicted_letter,
            'confidence': confidence,
            'gesture': gesture, # Sertakan gesture symbol di response
            'error': None
        })

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)