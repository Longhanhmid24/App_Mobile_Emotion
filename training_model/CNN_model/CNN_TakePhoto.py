from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
from tkinter import Tk, filedialog

# Load model
model = load_model('D:/model/App_Mobile_Emotion/training_model/CNN_model/full_model_notEarlyStopPart2.keras')


with open('D:/model/App_Mobile_Emotion/training_model/CNN_model/generator_info.json', 'r') as f:
    generator_info = json.load(f)

# Lấy danh sách class_labels
class_labels = generator_info["train_generator"]["class_labels"]
index_to_class = {idx: label for idx, label in enumerate(class_labels)}

# Hộp thoại chọn file ảnh
root = Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title='Chọn ảnh để dự đoán',
    filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')]
)

if not img_path:
    print("❌ Không chọn ảnh nào! Kết thúc chương trình.")
    exit()

# Đọc ảnh
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if img is None:
    print("❌ Không đọc được ảnh! Kết thúc chương trình.")
    exit()

# Chuyển grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Nếu ảnh đúng 48x48 (ảnh khuôn mặt cắt sẵn)
if gray.shape == (48, 48):
    face_img = cv2.resize(gray, (56, 56))
    face_array = face_img / 255.0
    face_array = np.expand_dims(face_array, axis=(0, -1))

    y_pred_probs = model.predict(face_array, verbose=0)
    y_pred_class = np.argmax(y_pred_probs)
    class_name = index_to_class[y_pred_class]
    confidence = y_pred_probs[0][y_pred_class]

    print(f"📸 Dự đoán cảm xúc: {class_name} ({confidence*100:.2f}%)")
    cv2.putText(img, f"{class_name} ({confidence*100:.1f}%)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

else:
    # Dùng Haar Cascade phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("❌ Không phát hiện khuôn mặt nào!")
        exit()

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (56, 56))
        face_array = face_img / 255.0
        face_array = np.expand_dims(face_array, axis=(0, -1))

        y_pred_probs = model.predict(face_array, verbose=0)
        y_pred_class = np.argmax(y_pred_probs)
        class_name = index_to_class[y_pred_class]
        confidence = y_pred_probs[0][y_pred_class]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{class_name} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Hiển thị ảnh kết quả
cv2.imshow('Emotion Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
