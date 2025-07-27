import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';
import '../widgets/CustomDrawer.dart';
import 'Feedback.dart';

class ReportsPage extends StatefulWidget {
  ReportsPage({super.key});

  @override
  State<ReportsPage> createState() => _ReportsPageState();
}

class _ReportsPageState extends State<ReportsPage> {
  final _supabase = Supabase.instance.client;
  List<Map<String, dynamic>> reports = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    fetchReports();
  }

  Future<void> fetchReports() async {
    final userId = _supabase.auth.currentUser?.id;
    if (userId == null) return;

    final data = await _supabase
        .from('Report')
        .select()
        .eq('User_id', userId)
        .order('created_at', ascending: false);

    setState(() {
      reports = List<Map<String, dynamic>>.from(data);
      isLoading = false;
    });
  }

  Future<void> _deleteReport(dynamic id) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Report'),
        content: const Text('Are you sure you want to delete this report?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          TextButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
        ],
      ),
    );

    if (confirm == true) {
      await _supabase.from('Report').delete().eq('id', id);
      fetchReports();
    }
  }

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true;
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: isUserSignedIn,
        backgroundColor: Colors.transparent,
      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn),
      body: BackgroundWrapper(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: kToolbarHeight + 40),
                Text(
                  'Presentation Reports'.tr(),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    shadows: [
                      Shadow(
                        blurRadius: 3.0,
                        color: Colors.white54,
                        offset: Offset(0, 0),
                      ),
                    ],
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                isLoading
                    ? const CircularProgressIndicator()
                    : reports.isEmpty
                        ? Text(
                            'no_reports_yet'.tr(),
                            style: const TextStyle(color: Colors.white70, fontSize: 18),
                          )
                        : Expanded(
                            child: ListView.builder(
                              itemCount: reports.length,
                              itemBuilder: (context, index) {
                                final report = reports[index];
                                return _buildReportCard(
                                  title: 'Presentation ${index + 1}',
                                  date: _formatDate(report['created_at']),
                                  score: report['Overall_score']?.toStringAsFixed(2) ?? 'N/A',
                                  onTap: () => _openReport(context, report),
                                  onDelete: () => _deleteReport(report['id']),
                                );
                              },
                            ),
                          ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildReportCard({
    required String title,
    required String date,
    required String score,
    required VoidCallback onTap,
    required VoidCallback onDelete,
  }) {
    return Card(
      color: Colors.white.withOpacity(0.9),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.symmetric(vertical: 10),
      child: ListTile(
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text("Date: $date\nScore: $score"),
        leading: const Icon(Icons.insert_drive_file, color: Colors.blueAccent),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              icon: const Icon(Icons.delete, color: Colors.redAccent),
              onPressed: onDelete,
            ),
            const Icon(Icons.arrow_forward_ios, size: 18),
          ],
        ),
        onTap: onTap,
      ),
    );
  }

  void _openReport(BuildContext context, Map<String, dynamic> report) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => FeedbackReportPage(report: report)),
    );
  }

  String _formatDate(String? timestamp) {
    if (timestamp == null) return '';
    final date = DateTime.tryParse(timestamp);
    if (date == null) return '';
    return DateFormat.yMMMMd().format(date);
  }
}