import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# --- Load model và thông tin class ---
model = load_model('D:/model/App_Mobile_Emotion/training_model/CNN_model/full_model_notEarlyStopPart2.keras')


with open('D:/model/App_Mobile_Emotion/training_model/CNN_model/generator_info.json', 'r') as f:
    generator_info = json.load(f)
class_labels = generator_info["train_generator"]["class_labels"]
index_to_class = {i: label for i, label in enumerate(class_labels)}

# --- Hàm tiền xử lý ảnh khuôn mặt ---
def preprocess_face(gray_img, x, y, w, h):
    face = gray_img[y:y+h, x:x+w]
    face = cv2.equalizeHist(face)  # Tăng tương phản
    face_resized = cv2.resize(face, (56, 56))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=(0, -1))  # shape: (1, 56, 56, 1)

# --- Khởi tạo cascade và webcam ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được webcam!")
    exit()

print("🎥 Webcam đang chạy... Nhấn 'q' để thoát.")

# --- Vòng lặp chính ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lỗi khi đọc frame!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_input = preprocess_face(gray, x, y, w, h)
        predictions = model.predict(face_input, verbose=0)[0]

        # Hiển thị xác suất từng cảm xúc
        for i, prob in enumerate(predictions):
            label = f"{index_to_class[i]}: {prob*100:.2f}%"
            cv2.putText(frame, label, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Nhãn chính xác nhất
        best_idx = np.argmax(predictions)
        best_label = f"{index_to_class[best_idx]} ({predictions[best_idx]*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, best_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow('Emotion Detection - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
