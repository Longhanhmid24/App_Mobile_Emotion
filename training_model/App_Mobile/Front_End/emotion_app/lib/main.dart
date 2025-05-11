import 'package:flutter/material.dart';
// Import your page files
import './page/call_api/photo.dart';
import './page/call_api/video.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // WebView platform initialization has changed in newer versions
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion Detection App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  // Create instances of your page widgets - removed const
  final List<Widget> _pages = [PredictPage(), VideoPage()];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Emotion Detection App')),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.photo),
            label: 'Dự đoán ảnh',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.videocam),
            label: 'Video nhận diện',
          ),
        ],
      ),
    );
  }
}
