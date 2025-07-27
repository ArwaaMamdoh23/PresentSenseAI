import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:my_flutter_project/Screens/Feedback.dart';
import 'package:path/path.dart' as path;
import 'package:file_picker/file_picker.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:http/http.dart' as http;
import 'package:easy_localization/easy_localization.dart';
import 'package:http_parser/http_parser.dart';

import '../widgets/LanguageSwitcherIcon.dart';
import '../widgets/background_wrapper.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/CustomDrawer.dart';

class UploadVideoPage extends StatefulWidget {
  const UploadVideoPage({super.key});

  @override
  State<UploadVideoPage> createState() => _UploadVideoPageState();
}

class _UploadVideoPageState extends State<UploadVideoPage> {
  final _picker = ImagePicker();
  final _supabase = Supabase.instance.client;

  File? _videoFile;
  bool _isUploading = false;
  double _uploadProgress = 0;

  Future<void> _pickVideo(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickVideo(source: source, maxDuration: const Duration(minutes: 5));
      if (pickedFile == null) return;

      setState(() {
        _videoFile = File(pickedFile.path);
        _uploadProgress = 0;
      });

      await _uploadVideoToFlask();
    } catch (e) {
      _showError('Error selecting video: $e'.tr());
    }
  }

Future<void> _uploadVideoToFlask() async {
  if (_videoFile == null) return;

  try {
    setState(() {
      _isUploading = true;
      _uploadProgress = 0;
    });

    final uri = Uri.parse('http://10.0.2.2:5000/upload_video');
    final userId = _supabase.auth.currentUser?.id ?? '';

    final request = http.MultipartRequest('POST', uri)
      ..fields['user_id'] = userId
      ..files.add(await http.MultipartFile.fromPath(
        'video',
        _videoFile!.path,
        contentType: MediaType('video', 'mp4'),
      ));

    final streamed = await request.send();
    setState(() => _uploadProgress = 1.0);

    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      _showError('Upload failed: ${response.reasonPhrase}');
      return;
    }

    // ✅ Now fetch the inserted report from Supabase
    final latestReport = await _supabase
        .from('Report')
        .select()
        .eq('User_id', userId)
        .order('created_at', ascending: false)
        .limit(1)
        .maybeSingle();

        // print('Latest Report from Supabase: $latestReport');


    if (latestReport == null) {
      _showError('No report retrieved from Supabase.');
      return;
    }

    // ✅ Navigate to FeedbackReportPage with correct data
    if (!mounted) return;
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => FeedbackReportPage(report: latestReport)),
    );
  } catch (e) {
    _showError('Upload error: $e');
  } finally {
    if (!mounted) return;
    setState(() {
      _isUploading = false;
      _uploadProgress = 0;
    });
  }
}


  void _showError(String message) => _snack(message, Colors.red);

  void _snack(String msg, Color color) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: color),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(
        showSignIn: true,
        isUserSignedIn: true,
        backgroundColor: Colors.transparent,
        extraActions: const [Padding(padding: EdgeInsets.only(right: 8), child: LanguageSwitcherIcon())],
      ),
      drawer: CustomDrawer(isSignedIn: true),
      body: BackgroundWrapper(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Upload Presentation'.tr(),
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      shadows: [Shadow(blurRadius: 3, color: Colors.white54)])),
              const SizedBox(height: 20),
              if (_videoFile != null) ...[
                const Icon(Icons.video_library, size: 60),
                const SizedBox(height: 10),
                Text(path.basename(_videoFile!.path), textAlign: TextAlign.center),
                const SizedBox(height: 20),
              ],
              if (_isUploading) ...[
                LinearProgressIndicator(value: _uploadProgress),
                const SizedBox(height: 15),
                Text('Uploading: ${(_uploadProgress * 100).toStringAsFixed(1)}%', style: Theme.of(context).textTheme.bodyLarge),
                const SizedBox(height: 30),
              ],
              ElevatedButton.icon(
                onPressed: () => _pickVideo(ImageSource.gallery),
                icon: const Icon(Icons.photo_library),
                label: Text('Select from Gallery'.tr()),
                style: _btnStyle,
              ),
              ElevatedButton.icon(
                onPressed: () async {
                  FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.video);
                  if (result != null) {
                    setState(() => _videoFile = File(result.files.single.path!));
                    await _uploadVideoToFlask();
                  }
                },
                icon: const Icon(Icons.folder),
                label: Text('Select from Drive'.tr()),
                style: _btnStyle,
              ),
              ElevatedButton.icon(
                onPressed: _isUploading ? null : _uploadVideoToFlask,
                icon: _isUploading
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.cloud_upload),
                label: Text('Upload Video'.tr()),
                style: _btnStyle,
              ),
            ],
          ),
        ),
      ),
    );
  }

  ButtonStyle get _btnStyle => ElevatedButton.styleFrom(
        minimumSize: const Size(double.infinity, 50),
        backgroundColor: Colors.transparent,
        shadowColor: Colors.transparent,
      );
}
