import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'emotion_predictor.dart';

class CameraEmotionScreen extends StatefulWidget {
  const CameraEmotionScreen({super.key});

  @override
  State<CameraEmotionScreen> createState() => _CameraEmotionScreenState();
}

class _CameraEmotionScreenState extends State<CameraEmotionScreen> {
  CameraController? _controller;
  bool _isDetecting = false;
  String _emotion = '';
  late EmotionPredictor _predictor;
  bool _modelLoaded = false;
  int _lastInferenceTime = 0;
  List<double> _emotionProbs = [];

  @override
  void initState() {
    super.initState();
    _predictor = EmotionPredictor();
    _initialize();
  }

  Future<void> _initialize() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Vui lòng cấp quyền camera')),
        );
      }
      return;
    }

    await _predictor.loadModel();
    setState(() {
      _modelLoaded = true;
    });

    await initCamera();
  }

  Future<void> initCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _controller!.initialize();

    await _controller!.startImageStream((CameraImage image) async {
      final now = DateTime.now().millisecondsSinceEpoch;
      if (_isDetecting || !_modelLoaded || now - _lastInferenceTime < 800) {
        return;
      }

      _isDetecting = true;
      _lastInferenceTime = now;

      try {
        final processedImage = _convertYUV420ToImage(image);
        if (processedImage != null) {
          final output = await _predictor.predictRaw(processedImage);
          final label = await _predictor.predict(processedImage);
          if (mounted) {
            setState(() {
              _emotion = label;
              _emotionProbs = output;
            });
          }
        }
      } catch (e) {
        print('Lỗi xử lý ảnh hoặc dự đoán: $e');
      }

      _isDetecting = false;
    });

    if (mounted) setState(() {});
  }

  /// Chuyển CameraImage (YUV420) sang ảnh grayscale Image package, xử lý xoay ảnh camera front
  img.Image _convertYUV420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final img.Image grayscaleImage = img.Image(width: width, height: height);

    final yPlane = image.planes[0].bytes;
    final yRowStride = image.planes[0].bytesPerRow;
    final yPixelStride = image.planes[0].bytesPerPixel ?? 1;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yIndex = y * yRowStride + x * yPixelStride;
        if (yIndex < yPlane.length) {
          final pixelValue = yPlane[yIndex];
          grayscaleImage.setPixelRgba(
            x,
            y,
            pixelValue,
            pixelValue,
            pixelValue,
            255,
          );
        }
      }
    }

    // Lật ngang (mirror) để khớp với camera front
    final img.Image flipped = img.flipHorizontal(grayscaleImage);

    // Xoay 90 độ cho ảnh đúng chiều
    final img.Image rotated = img.copyRotate(flipped, angle: 90);

    return rotated;
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body:
          _controller == null || !_controller!.value.isInitialized
              ? const Center(child: CircularProgressIndicator())
              : Stack(
                children: [
                  CameraPreview(_controller!),
                  Positioned(
                    top: 40,
                    left: 20,
                    child: Text(
                      'Cảm xúc: $_emotion',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        shadows: [
                          Shadow(
                            color: Colors.black54,
                            offset: Offset(1, 1),
                            blurRadius: 3,
                          ),
                        ],
                      ),
                    ),
                  ),
                  Positioned(
                    bottom: 20,
                    left: 20,
                    right: 20,
                    child:
                        _emotionProbs.isEmpty
                            ? const SizedBox()
                            : Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: List.generate(_emotionProbs.length, (
                                i,
                              ) {
                                final label = EmotionPredictor.labels[i];
                                final percent = (_emotionProbs[i] * 100)
                                    .toStringAsFixed(1);
                                return Text(
                                  '$label: $percent%',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 16,
                                    shadows: [
                                      Shadow(
                                        color: Colors.black87,
                                        offset: Offset(1, 1),
                                        blurRadius: 2,
                                      ),
                                    ],
                                  ),
                                );
                              }),
                            ),
                  ),
                ],
              ),
    );
  }
}
