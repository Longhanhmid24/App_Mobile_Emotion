import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class PredictPage extends StatefulWidget {
  @override
  _PredictPageState createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  File? _image;
  String _result = '';

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    setState(() => _image = File(pickedFile.path));
    await _sendImage(File(pickedFile.path));
  }

  Future<void> _sendImage(File image) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://10.0.2.2:5000/predict'),
    );
    request.files.add(await http.MultipartFile.fromPath('image', image.path));
    final response = await request.send();
    final responseData = await response.stream.bytesToString();
    final decoded = jsonDecode(responseData);

    setState(() {
      _result = decoded['top_prediction'] ?? 'Không rõ';
    });
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: Text('Dự đoán khuôn mặt')),
    body: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _image != null ? Image.file(_image!) : Container(),
        ElevatedButton(onPressed: _pickImage, child: Text('Chọn ảnh')),
        SizedBox(height: 20),
        Text('Kết quả: $_result'),
      ],
    ),
  );
}
