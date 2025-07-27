import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:my_flutter_project/Screens/AboutUs.dart';
import 'package:my_flutter_project/Screens/Instructions.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'Screens/AuthPage.dart';
import 'Screens/SignUpPage.dart';
import 'Screens/SignInPage.dart';
import 'Screens/HomePage.dart';
import 'Screens/SettingsPage.dart';
import 'Screens/ProfilePage.dart';
import 'Screens/ReportsPage.dart';
import 'Screens/UploadVideo.dart';


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await EasyLocalization.ensureInitialized();

  await Supabase.initialize(
    url: 'https://ohllbliwedftnyqmthze.supabase.co',
    anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obGxibGl3ZWRmdG55cW10aHplIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM1NDIxMzAsImV4cCI6MjA1OTExODEzMH0.XW1XNf7v3-JX94-1xJNgPM70t2qvZoEClyAab85ie1o',
  );

  runApp(
    EasyLocalization(
      supportedLocales: const [
        Locale('en'),
        Locale('ar'),
        Locale('fr'),
        Locale('zh'),
        Locale('nl'),
      ],
      path: 'assets/translation',
      fallbackLocale: const Locale('en'),
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PresentSense',
      debugShowCheckedModeBanner: false,
      locale: context.locale,
      supportedLocales: context.supportedLocales,
      localizationsDelegates: context.localizationDelegates,
      initialRoute: '/home',
      routes: {
        '/home': (context) => HomePage(),
        '/auth': (context) => AuthPage(),
        '/sign-up': (context) => SignUpPage(),
        '/sign-in': (context) => SignInPage(),
        '/settings': (context) => SettingsPage(),
        '/profile': (context) => ProfilePage(),
        '/report': (context) => ReportsPage(),
        '/upload': (context) => UploadVideoPage(),
        '/aboutus': (context) => AboutUs(),
        '/instructions': (context) => Instructions(),
        
      },
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue.shade400,
          primary: Colors.blue.shade500,
          secondary: Colors.grey.shade300,
        ),
        scaffoldBackgroundColor: Colors.white,
        textTheme: const TextTheme(
          headlineLarge: TextStyle(color: Colors.black, fontSize: 32, fontWeight: FontWeight.bold),
          bodyLarge: TextStyle(color: Colors.black),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue.shade600,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30),
            ),
            padding: const EdgeInsets.symmetric(vertical: 15),
          ),
        ),
      ),
    );
  }
}
