import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# --- Tải mô hình đã huấn luyện ---
model = load_model('D:/model/App_Mobile_Emotion/training_model/CNN_model/CNN_Model_FER2013.keras')

# --- Danh sách nhãn cảm xúc ---
class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Hàm xử lý ảnh khuôn mặt ---
def preprocess_face(gray_img, x, y, w, h):
    face = gray_img[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # shape: (1, 48, 48, 1)
    return face

# --- Khởi tạo cascade và webcam ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được webcam!")
    exit()

# --- Khởi tạo bộ CLAHE (tăng tương phản cục bộ) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

print("🎥 Webcam đang chạy... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame!")
        break

    # --- Chuyển sang ảnh xám và tăng tương phản ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe.apply(gray)  # Dùng CLAHE thay vì histogram thường

    faces = face_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_input = preprocess_face(enhanced_gray, x, y, w, h)
        predictions = model.predict(face_input, verbose=0)[0]

        # Vẽ khung và hiển thị nhãn cảm xúc
        best_idx = np.argmax(predictions)
        best_label = f"{class_labels[best_idx]} ({predictions[best_idx]*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, best_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Hiển thị xác suất từng lớp (tùy chọn)
        for i, prob in enumerate(predictions):
            text = f"{class_labels[i]}: {prob*100:.2f}%"
            cv2.putText(frame, text, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow('WebCam Mo Hinh CNN - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
