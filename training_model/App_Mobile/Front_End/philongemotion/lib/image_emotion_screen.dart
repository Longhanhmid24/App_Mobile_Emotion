import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'emotion_predictor.dart';

class ImageEmotionScreen extends StatefulWidget {
  const ImageEmotionScreen({super.key});

  @override
  State<ImageEmotionScreen> createState() => _ImageEmotionScreenState();
}

class _ImageEmotionScreenState extends State<ImageEmotionScreen> {
  String _emotion = '';
  img.Image? _displayedImage;
  late EmotionPredictor _predictor;
  bool _modelLoaded = false;

  @override
  void initState() {
    super.initState();
    _predictor = EmotionPredictor();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _predictor.loadModel();
    setState(() {
      _modelLoaded = true;
    });
  }

  Future<void> _pickImageAndPredict() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      final original = img.decodeImage(bytes);

      if (original == null) return;

      final resized = img.copyResize(original, width: 48, height: 48);
      final grayscale = img.grayscale(resized);

      final label = await _predictor.predict(grayscale);

      setState(() {
        _emotion = label;
        _displayedImage = original;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Nhận diện từ ảnh')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            ElevatedButton.icon(
              onPressed: _modelLoaded ? _pickImageAndPredict : null,
              icon: const Icon(Icons.image_search),
              label: const Text("Chọn ảnh từ thư viện"),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(double.infinity, 50),
              ),
            ),
            const SizedBox(height: 20),
            if (_displayedImage != null)
              Card(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 4,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    Uint8List.fromList(img.encodeJpg(_displayedImage!)),
                    height: 250,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
            const SizedBox(height: 20),
            Text(
              _emotion.isNotEmpty ? 'Cảm xúc: $_emotion' : 'Chưa có kết quả',
              style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w500),
            ),
          ],
        ),
      ),
    );
  }
}
