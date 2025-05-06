from flask import Flask, Response
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model & labels
model = load_model('C:/Users/DELL/Desktop/model/training_model/CNN_model/full_model_notEarlyStopPart2.keras')
with open('C:/Users/DELL/Desktop/model/training_model/CNN_model/generator_info.json', 'r') as f:
    generator_info = json.load(f)
class_labels = generator_info["train_generator"]["class_labels"]
index_to_class = {i: label for i, label in enumerate(class_labels)}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def preprocess_face(gray_img, x, y, w, h):
    face = gray_img[y:y+h, x:x+w]
    face = cv2.equalizeHist(face)
    face_resized = cv2.resize(face, (56, 56))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=(0, -1))

app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_input = preprocess_face(gray, x, y, w, h)
            predictions = model.predict(face_input, verbose=0)[0]

            best_idx = np.argmax(predictions)
            best_label = f"{index_to_class[best_idx]} ({predictions[best_idx]*100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, best_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
