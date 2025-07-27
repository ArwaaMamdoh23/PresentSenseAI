import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:easy_localization/easy_localization.dart';  // Import easy_localization

class CustomDrawer extends StatelessWidget {
  final bool isSignedIn;

  const CustomDrawer({super.key, required this.isSignedIn});

  @override
  Widget build(BuildContext context) {
    return Drawer(
      
      child: SafeArea(
        child: Column(
          children: [
            ListTile(
              leading: const Icon(Icons.person),
              title: Text('Profile'.tr()), // Localize text
              onTap: () {
                Navigator.pushNamed(context, '/profile');
              },
            ),
            ListTile(
              leading: const Icon(Icons.upload_file),
              title: Text('Upload'.tr()), // Localize text
              onTap: () {
                Navigator.pushNamed(context, '/upload');
              },
            ),
            ListTile(
              leading: const Icon(Icons.insert_drive_file),
              title: Text('Reports'.tr()), // Localize text
              onTap: () {
                Navigator.pushNamed(context, '/report');
              },
            ),
            ListTile(
              leading: const Icon(Icons.help),
              title: Text('Instructions'.tr()), // Localize text
              onTap: () {
                Navigator.pushNamed(context, '/instructions');
              },
            ),
            // ListTile(
            //   leading: const Icon(Icons.help),
            //   title: Text('Loading'.tr()), // Localize text
            //   onTap: () {
            //     Navigator.pushNamed(context, '/loading');
            //   },
            // ),
            const Spacer(),
            ListTile(
              leading: const Icon(Icons.settings),
              title: Text('Settings'.tr()), // Localize text
              onTap: () {
                Navigator.pushNamed(context, '/settings');
              },
            ),
            ListTile(
              leading: const Icon(Icons.info),
              title: Text('About Us'.tr()), // Localize text
              onTap: () {
                Navigator.pushNamed(context, '/aboutus');
              },
            ),
            if (isSignedIn)
              Align(
                alignment: Alignment.bottomCenter,
                child: ListTile(
                  leading: const Icon(Icons.exit_to_app, color: Colors.red),
                  title: Text(
                    'Log Out'.tr(),  // Localize text
                    style: const TextStyle(color: Colors.red),
                  ),
                  onTap: () async {
                    try {
                      await Supabase.instance.client.auth.signOut();
                      Navigator.pushNamedAndRemoveUntil(
                          context, '/home', (route) => false);
                    } catch (e) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Logout failed: $e')),
                      );
                    }
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}
