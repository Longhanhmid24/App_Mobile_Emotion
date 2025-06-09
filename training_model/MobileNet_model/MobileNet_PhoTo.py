import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk

# --- 1. Load mô hình MobileNet ---
model = load_model('C:/Tin/DACS_App_Mobile_Emotion/App_Mobile_Emotion/training_model/MobileNet_model/MobileNet_Model.keras')

# --- 2. Nhãn cảm xúc ---
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- 3. Hàm xử lý khuôn mặt ---
def preprocess_face(face):
    face = cv2.resize(face, (96, 96))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# --- 4. Chọn ảnh từ máy ---
def choose_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    return file_path

# --- 5. Điều chỉnh sáng và tương phản ---
def adjust_brightness_contrast(image, brightness=10, contrast=50):
    alpha = contrast / 127 + 1.0
    beta = brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# --- 6. Dự đoán cảm xúc từ ảnh ---
def predict_emotion_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Không thể đọc ảnh.")
        return

    # Resize nếu ảnh quá lớn
    max_width = 800
    if img.shape[1] > max_width:
        scale_ratio = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)

    # Tăng sáng/contrast + chuyển ảnh xám để phát hiện khuôn mặt
    enhanced = adjust_brightness_contrast(img)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Phát hiện khuôn mặt
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80), maxSize=(400, 400))

    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt.")
        return

    print(f"✅ Phát hiện {len(faces)} khuôn mặt.")

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        face_processed = preprocess_face(face_img)
        preds = model.predict(face_processed, verbose=0)[0]
        label_idx = np.argmax(preds)
        label = f"{emotion_labels[label_idx]} ({preds[label_idx]*100:.1f}%)"

        # Hiển thị các nhãn xác suất
        bar_height = 30
        spacing = 10
        bar_width_max = 200
        margin = 20
        extra_space = (bar_height + spacing) * len(emotion_labels) + margin * 2

        new_img = np.ones((img.shape[0] + extra_space, img.shape[1], 3), dtype=np.uint8) * 255
        new_img[:img.shape[0], :, :] = img

        for i, prob in enumerate(preds):
            text = f"{emotion_labels[i]}: {prob*100:.2f}%"
            color = (0, 128, 0) if i == label_idx else (120, 120, 120)
            y_offset = img.shape[0] + margin + i * (bar_height + spacing)

            cv2.putText(new_img, text, (10, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bar_len = int(prob * bar_width_max)
            cv2.rectangle(new_img, (180, y_offset), (180 + bar_len, y_offset + bar_height - 5), color, -1)
            cv2.rectangle(new_img, (180, y_offset), (180 + bar_width_max, y_offset + bar_height - 5), (180, 180, 180), 2)

        # Vẽ khung và nhãn khuôn mặt
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(new_img, label, (x -30, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Hiển thị ảnh kết quả
        cv2.imshow("Emotion MobileNet - Static Image", cv2.resize(new_img, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- 7. Chạy chương trình ---
if __name__ == "__main__":
    img_path = choose_image()
    if img_path:
        predict_emotion_from_image(img_path)
    else:
        print("❌ Không có ảnh nào được chọn.")
