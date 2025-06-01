import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. Load mô hình ---
model = load_model('D:/model/App_Mobile_Emotion/training_model/MobileNet_model/MobileNet_Model.keras')

# --- 2. Nhãn cảm xúc ---
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- 3. Hàm xử lý khuôn mặt ---
def preprocess_face(face):
    face = cv2.resize(face, (96, 96))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# --- 4. Điều chỉnh sáng và tương phản ---
def adjust_brightness_contrast(image, brightness=10, contrast=50):  # ↓ giảm brightness
    alpha = contrast / 127 + 1.0
    beta = brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# --- 5. Khởi tạo detector và CLAHE ---
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --- 6. Mở camera ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Điều chỉnh sáng/contrast
    frame = adjust_brightness_contrast(frame, brightness=10, contrast=50)

    # Chuyển ảnh xám, cân bằng sáng & làm mượt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Phát hiện khuôn mặt với tham số tối ưu
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80),
        maxSize=(400, 400)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        face_processed = preprocess_face(face_img)
        preds = model.predict(face_processed)[0]
        label_idx = np.argmax(preds)
        label = f"{emotion_labels[label_idx]} ({preds[label_idx]*100:.1f}%)"

        # Hiển thị từng xác suất cảm xúc
        for i, emo in enumerate(emotion_labels):
            text = f"{emo}: {preds[i]*100:.2f}%"
            cv2.putText(frame, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Vẽ khung và nhãn
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Hiển thị kết quả
    cv2.imshow("Emotion MobileNet - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Giải phóng ---
cap.release()
cv2.destroyAllWindows()
