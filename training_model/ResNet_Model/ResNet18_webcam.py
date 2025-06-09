import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Cấu hình ---
IMG_SIZE = 96
emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']
model = load_model('C:/Tin/DACS_App_Mobile_Emotion/App_Mobile_Emotion/training_model/ResNet_Model/resnet18_model.keras')

# --- Hàm xử lý ảnh xám nâng cao ---
def enhance_gray_image(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

# --- Webcam ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("[INFO] Đang mở camera... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = enhance_gray_image(gray)

    # Cải thiện phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(
        enhanced_gray,
        scaleFactor=1.1,
        minNeighbors=6,         # tăng để giảm sai lệch
        minSize=(80, 80),       # loại bỏ vùng nhỏ
        maxSize=(400, 400)      # tránh nhận nhầm vùng lớn
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_rgb = np.stack((roi_resized,) * 3, axis=-1) / 255.0
        roi_rgb = np.expand_dims(roi_rgb, axis=0)

        predictions = model.predict(roi_rgb, verbose=0)[0]
        best_idx = np.argmax(predictions)
        best_label = f"{emotion_labels[best_idx]} ({predictions[best_idx]*100:.2f}%)"

        # Vẽ khung mặt + nhãn chính
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, best_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # In xác suất của tất cả cảm xúc
        for i, (label, prob) in enumerate(zip(emotion_labels, predictions)):
            text = f"{label}: {prob * 100:.2f}%"
            cv2.putText(frame, text, (x + w + 10, y + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Emotion Detection - ResNet18", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
