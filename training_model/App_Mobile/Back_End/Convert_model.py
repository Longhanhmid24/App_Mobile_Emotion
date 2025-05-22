import tensorflow as tf
from tensorflow.keras.models import load_model

# Load mô hình đã huấn luyện
model = load_model('D:/model/App_Mobile_Emotion/training_model/CNN_model/Model_CNN.keras')

# Convert sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Tùy chọn) Nếu muốn giảm dung lượng model:
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert và lưu
tflite_model = converter.convert()

# Ghi ra file
with open("Model_CNN.tflite", "wb") as f:
    f.write(tflite_model)
    
print("✅ File TFLite đã được ghi thành công.")