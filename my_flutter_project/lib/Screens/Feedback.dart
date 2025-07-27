import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/CustomDrawer.dart';

class FeedbackReportPage extends StatefulWidget {
  final Map<String, dynamic> report;
  const FeedbackReportPage({super.key, required this.report});

  @override
  State<FeedbackReportPage> createState() => _FeedbackReportPageState();
}

class _FeedbackReportPageState extends State<FeedbackReportPage> {
  Map<String, dynamic> get report => widget.report;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: true,
        backgroundColor: const Color.fromARGB(197, 185, 185, 185),
      ),
      drawer: CustomDrawer(isSignedIn: true),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          buildSection('Comprehensive Feedback', [
            'dominant_emotion',
            'dominant_eye_contact',
            'emotion_feedback',
          ]),
          buildSection('Posture Analysis', [
            'dominant_posture',
            // 'posture_meaning',
            'posture_feedback',
          ]),
          buildSection('Gesture Analysis', [
            'dominant_gesture_1',
            'gesture_1_meaning',
            'dominant_gesture_2',
            'gesture_2_meaning',
            'gesture_feedback',
          ]),
          buildSection('Speech Analysis', [
            'detected_language',
          ]),
          buildSection('Grammar Analysis', [
            'grammar_score',
            'grammar_feedback',
          ]),
          buildSection('Speech Pace Analysis', [
            'speech_pace',
            // 'pace_score',
            'pace_feedback',
          ]),
          buildSection('Fluency Analysis', [
            'fluency_score',
            // 'filler_words',
            'fluency_feedback',
          ]),
          buildSection('Pronunciation Analysis', [
            'pronunciation_score',
            'pronunciation_feedback',
          ]),
          const SizedBox(height: 30),
          Center(
            child: Text(
              'Overall Score: ${report['Overall_score'] != null ? report['Overall_score'].toString() : 'N/A'}',
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.green,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget buildSection(String title, List<String> keys) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        sectionTitle(title),
        const SizedBox(height: 8),
        ...keys.map((key) => infoRow(
              _prettifyKey(key),
              formatValue(report[key]),
            )),
        const SizedBox(height: 20),
      ],
    );
  }

  String formatValue(dynamic value) {
    if (value == null) return '';
    try {
      if (value is Map) {
        return value.entries.map((e) => "${e.key}: ${e.value}").join(', ');
      }
      return value.toString();
    } catch (_) {
      return '';
    }
  }

  String _prettifyKey(String key) {
    return key
        .replaceAll('_', ' ')
        .splitMapJoin(RegExp(r'[A-Z]'),
            onMatch: (m) => ' ${m.group(0)}', onNonMatch: (n) => n)
        .capitalize();
  }

  Widget sectionTitle(String title) => Text(
        title.tr(),
        style: const TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          color: Colors.green,
        ),
      );

  Widget infoRow(String label, dynamic value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4.0),
        child: RichText(
          text: TextSpan(
            text: '$label: ',
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
            children: [
              TextSpan(
                text: value != null ? value.toString() : '',
                style: const TextStyle(fontWeight: FontWeight.normal),
              ),
            ],
          ),
        ),
      );
}

extension StringCasingExtension on String {
  String capitalize() =>
      isNotEmpty ? '${this[0].toUpperCase()}${substring(1)}' : this;
}
