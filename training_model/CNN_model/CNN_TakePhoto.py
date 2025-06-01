import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk

# --- Tải model đã huấn luyện ---
model = load_model('D:/model/App_Mobile_Emotion/training_model/CNN_model/CNN_Model_FER2013.keras')

# --- Nhãn cảm xúc ---
class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Hàm xử lý khuôn mặt ---
def preprocess_face(gray_img, x, y, w, h):
    face = gray_img[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)  # (48, 48, 1)
    face = np.expand_dims(face, axis=0)   # (1, 48, 48, 1)
    return face

# --- Chọn ảnh từ máy ---
def choose_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    return file_path

# --- Dự đoán cảm xúc từ ảnh ---
def predict_emotion_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Không thể đọc ảnh.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tăng cường ảnh bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt nào.")
        return

    for (x, y, w, h) in faces:
        face_input = preprocess_face(enhanced_gray, x, y, w, h)
        predictions = model.predict(face_input, verbose=0)[0]

        best_idx = int(np.argmax(predictions))
        best_label = f"{class_labels[best_idx]}: {predictions[best_idx]*100:.2f}%"

        # Vẽ khung khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ⚠️ Đưa nhãn sang bên trái
        cv2.putText(img, best_label, (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Thêm không gian bên dưới để vẽ thanh cảm xúc ---
        bar_height = 30
        spacing = 10
        bar_width_max = 200  # ⚠️ Rút ngắn chiều dài thanh biểu đồ
        margin = 20
        extra_space = (bar_height + spacing) * len(class_labels) + margin * 2

        # Tạo ảnh mới có thêm không gian trắng bên dưới
        new_img = np.ones((img.shape[0] + extra_space, img.shape[1], 3), dtype=np.uint8) * 255
        new_img[:img.shape[0], :, :] = img

        # --- Vẽ biểu đồ cảm xúc bên dưới ảnh ---
        for i, prob in enumerate(predictions):
            label = f"{class_labels[i]}: {prob * 100:.2f}%"
            color = (0, 128, 0) if i == best_idx else (100, 100, 100)
            y_offset = img.shape[0] + margin + i * (bar_height + spacing)

            # Text bên trái
            cv2.putText(new_img, label, (10, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Thanh biểu đồ cảm xúc
            bar_length = int(prob * bar_width_max)
            cv2.rectangle(new_img, (180, y_offset), (180 + bar_length, y_offset + bar_height - 5), color, -1)
            cv2.rectangle(new_img, (180, y_offset), (180 + bar_width_max, y_offset + bar_height - 5), (150, 150, 150), 2)

        # Hiển thị ảnh
        cv2.imshow("Nhan Dien Anh Tinh CNN", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

# --- Chạy chương trình ---
if __name__ == "__main__":
    img_path = choose_image()
    if img_path:
        predict_emotion_from_image(img_path)
    else:
        print("❌ Không có ảnh nào được chọn.")
