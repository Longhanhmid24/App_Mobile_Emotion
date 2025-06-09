from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model & labels
model = load_model('C:/Tin/DACS_App_Mobile_Emotion/App_Mobile_Emotion/training_model/CNN_model/full_model_notEarlyStopPart2.keras')
with open('C:/Tin/DACS_App_Mobile_Emotion/App_Mobile_Emotion/training_model/CNN_model/generator_info.json', 'r') as f:
    generator_info = json.load(f)
class_labels = generator_info["train_generator"]["class_labels"]
index_to_class = {i: label for i, label in enumerate(class_labels)}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(gray_img, x, y, w, h):
    face = gray_img[y:y+h, x:x+w]
    face = cv2.equalizeHist(face)
    face_resized = cv2.resize(face, (56, 56))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=(0, -1))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_data = request.files['image'].read()
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 200

    x, y, w, h = faces[0]
    face_input = preprocess_face(img, x, y, w, h)
    preds = model.predict(face_input, verbose=0)[0]

    results = {index_to_class[i]: float(f"{pred:.4f}") for i, pred in enumerate(preds)}
    top_pred = index_to_class[np.argmax(preds)]
    confidence = float(f"{np.max(preds):.4f}")

    return jsonify({
        'top_prediction': top_pred,
        'confidence': confidence,
        'all_predictions': results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
