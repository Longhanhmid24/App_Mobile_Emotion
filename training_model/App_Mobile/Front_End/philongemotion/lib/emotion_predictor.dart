import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class EmotionPredictor {
  late Interpreter _interpreter;

  // Danh sách nhãn cảm xúc public
  static const List<String> labels = [
    "Tức giận", // 0: Angry
    "Sợ hãi", // 2: Fear
    "Vui vẻ", // 3: Happy
    "Bình thường", // 4: Neutral
    "Buồn", // 5: Sad
    "Ngạc nhiên", // 6: Surprise
  ];

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(
      'lib/assets/model/Model_CNN.tflite',
      options: InterpreterOptions()..threads = 2,
    );
  }

  /// Dự đoán cảm xúc từ ảnh khuôn mặt 48x48, trả về label
  Future<String> predict(img.Image image) async {
    final resizedImage = img.copyResize(image, width: 48, height: 48);
    final grayscaleImage = img.grayscale(resizedImage);

    final input = List.generate(
      1,
      (_) => List.generate(
        48,
        (i) => List.generate(48, (j) {
          final pixel = grayscaleImage.getPixel(j, i);
          final luminance = img.getLuminance(pixel);
          return [luminance / 255.0];
        }),
      ),
    );

    final output = List.generate(1, (_) => List.filled(6, 0.0));

    _interpreter.run(input, output);

    final prediction = output[0];
    final maxIndex = prediction.indexOf(
      prediction.reduce((a, b) => a > b ? a : b),
    );

    return labels[maxIndex];
  }

  /// Trả về vector xác suất các cảm xúc
  Future<List<double>> predictRaw(img.Image image) async {
    final resizedImage = img.copyResize(image, width: 48, height: 48);
    final grayscaleImage = img.grayscale(resizedImage);

    final input = List.generate(
      1,
      (_) => List.generate(
        48,
        (i) => List.generate(48, (j) {
          final pixel = grayscaleImage.getPixel(j, i);
          final luminance = img.getLuminance(pixel);
          return [luminance / 255.0];
        }),
      ),
    );

    final output = List.generate(1, (_) => List.filled(6, 0.0));

    _interpreter.run(input, output);

    return output[0];
  }
}
