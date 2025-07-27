import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:my_flutter_project/Screens/HomePage.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';
import 'package:my_flutter_project/Screens/SignInPage.dart';
import '../widgets/CustomDrawer.dart'; 

class SettingsPage extends StatelessWidget {
  SettingsPage({super.key});
  final _supabase = Supabase.instance.client;
@override
Widget build(BuildContext context) {
  return Scaffold(
    extendBodyBehindAppBar: true,
    appBar: CustomAppBar(
      showSignIn: false,
      isUserSignedIn: true,
        backgroundColor: Colors.transparent,  // Makes the app bar transparent
    ),
    drawer: CustomDrawer(isSignedIn: true),
    body: BackgroundWrapper(
      child: Column(
        children: [
          // Add padding to push the "Settings" text below the AppBar
          const SizedBox(height: kToolbarHeight + 40),  // Add more padding for extra space below AppBar
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 20),
            child: Text(
              "Settings",
              style: TextStyle(
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
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildSettingsItem(
                    icon: Icons.dark_mode,
                    title: "Theme",
                    subtitle: "Switch between Light and Dark mode",
                    onTap: () {},
                  ),
                  _buildSettingsItem(
                    icon: Icons.notifications,
                    title: "Notifications",
                    subtitle: "Manage notification preferences",
                    onTap: () {},
                  ),
                  _buildSettingsItem(
                    icon: Icons.privacy_tip,
                    title: "Terms & Privacy",
                    subtitle: "View our terms and privacy policy",
                    onTap: () {},
                  ),
                  _buildSettingsItem(
                    icon: Icons.delete_forever,
                    title: "Delete Account",
                    subtitle: "Permanently remove your account",
                    onTap: () {
                      _showDeleteConfirmationDialog(context);
                    },
                  ),
                  _buildSettingsItem(
                    icon: Icons.logout,
                    title: "Sign Out",
                    subtitle: "Log out from your account",
                    onTap: () {
                      _signOut(context);
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    ),
  );
}

  Widget _buildSettingsItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Card(
      color: Colors.grey[900]?.withOpacity(0.8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListTile(
        leading: Icon(icon, color: Colors.white),
        title: Text(
          title,
          style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
        ),
        subtitle: Text(subtitle, style: const TextStyle(color: Colors.grey)),
        trailing: const Icon(Icons.arrow_forward_ios, color: Colors.white),
        onTap: onTap,
      ),
    );
  }

  void _signOut(BuildContext context) async {
    await _supabase.auth.signOut();
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const HomePage()),
    );
  }

  void _showDeleteConfirmationDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("Confirm Deletion"),
          content: const Text("Are you sure you want to permanently delete your account? This action cannot be undone."),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text("Cancel", style: TextStyle(color: Colors.blue)),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                _deleteAccount(context);
              },
              child: const Text("Delete", style: TextStyle(color: Colors.red)),
            ),
          ],
        );
      },
    );
  }

  Future<String?> _askForPassword(BuildContext context) async {
    TextEditingController passwordController = TextEditingController();

    return await showDialog<String>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("Confirm Password"),
          content: TextField(
            controller: passwordController,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: "Enter your password",
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, null),
              child: const Text("Cancel"),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context, passwordController.text.trim());
              },
              child: const Text("Confirm"),
            ),
          ],
        );
      },
    );
  }

  void _deleteAccount(BuildContext context) async {
    final user = _supabase.auth.currentUser;
    if (user == null) return;

    try {
      if (user.email != null) {
        String? currentPassword = await _askForPassword(context);
        if (currentPassword == null || currentPassword.isEmpty) return;

        final authResponse = await _supabase.auth.signInWithPassword(
          email: user.email!,
          password: currentPassword,
        );

        if (authResponse.user == null) {
          throw Exception("Reauthentication failed");
        }
      }

      final existingRow = await _supabase
          .from('User')
          .select()
          .eq('User_id', user.id)
          .maybeSingle();

      if (existingRow != null) {
        await _supabase.from('User').delete().eq('User_id', user.id);
        print(" User deleted from 'User' table");
      } else {
        print("No matching row found in User table for deletion.");
      }

      await _supabase.auth.admin.deleteUser(user.id);
      print("Deleted from Supabase Auth");

      await _supabase.auth.signOut();

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Account deleted successfully")),
        );
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (context) => const SignInPage()),
          (route) => false,
        );
      }
    } catch (e) {
      print("Error deleting account: $e");
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error deleting account: \${e.toString()}")),
        );
      }
    }
  }
}