import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart'; // Thêm import này

class VideoPage extends StatefulWidget {
  const VideoPage({Key? key}) : super(key: key);

  @override
  State<VideoPage> createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  late final WebViewController controller;

  @override
  void initState() {
    super.initState();
    // Khởi tạo controller cho WebView
    controller =
        WebViewController()
          ..setJavaScriptMode(JavaScriptMode.unrestricted)
          ..loadRequest(Uri.parse('http://10.0.2.2:5001/video_feed'));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Live Video Feed')),
      body: WebViewWidget(controller: controller),
    );
  }
}
