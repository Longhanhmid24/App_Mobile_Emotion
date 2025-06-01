import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk

# --- Tải model ResNet ---
model = load_model('D:/model/App_Mobile_Emotion/training_model/ResNet_Model/resnet18_model.keras')

# --- Nhãn cảm xúc ---
class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Hàm xử lý khuôn mặt ---
def preprocess_face(color_img, x, y, w, h):
    face = color_img[y:y+h, x:x+w]
    face = cv2.resize(face, (96, 96))  # Đảm bảo đúng kích thước input model
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)  # (1, 96, 96, 3)
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

    # Resize ảnh nếu quá lớn để cải thiện nhận diện khuôn mặt
    max_width = 800
    if img.shape[1] > max_width:
        scale_ratio = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tăng cường ảnh xám bằng CLAHE (chỉ dùng để phát hiện khuôn mặt)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt nào.")
        return

    print(f"✅ Phát hiện {len(faces)} khuôn mặt.")

    for (x, y, w, h) in faces:
        face_input = preprocess_face(img, x, y, w, h)  # Dùng ảnh màu gốc
        predictions = model.predict(face_input, verbose=0)[0]

        best_idx = int(np.argmax(predictions))
        best_label = f"{class_labels[best_idx]}: {predictions[best_idx]*100:.2f}%"

        # Vẽ khung và nhãn
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, best_label, (x -30 , y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Thêm không gian bên dưới để hiển thị biểu đồ cảm xúc
        bar_height = 30
        spacing = 10
        bar_width_max = 200
        margin = 20
        extra_space = (bar_height + spacing) * len(class_labels) + margin * 2

        new_img = np.ones((img.shape[0] + extra_space, img.shape[1], 3), dtype=np.uint8) * 255
        new_img[:img.shape[0], :, :] = img

        # Vẽ biểu đồ cảm xúc
        for i, prob in enumerate(predictions):
            label = f"{class_labels[i]}: {prob * 100:.2f}%"
            color = (0, 128, 0) if i == best_idx else (100, 100, 100)
            y_offset = img.shape[0] + margin + i * (bar_height + spacing)

            cv2.putText(new_img, label, (10, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bar_length = int(prob * bar_width_max)
            cv2.rectangle(new_img, (180, y_offset), (180 + bar_length, y_offset + bar_height - 5), color, -1)
            cv2.rectangle(new_img, (180, y_offset), (180 + bar_width_max, y_offset + bar_height - 5), (150, 150, 150), 2)

        # Hiển thị kết quả
        cv2.imshow("Nhan Dien Cam Xuc - ResNet", cv2.resize(new_img, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Chạy chương trình ---
if __name__ == "__main__":
    img_path = choose_image()
    if img_path:
        predict_emotion_from_image(img_path)
    else:
        print("❌ Không có ảnh nào được chọn.")
