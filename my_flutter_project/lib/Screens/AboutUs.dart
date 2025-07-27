import 'package:flutter/material.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';
import '../widgets/CustomDrawer.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:easy_localization/easy_localization.dart'; // Add this import

class AboutUs extends StatelessWidget {
  const AboutUs({super.key});

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true;
    double screenWidth = MediaQuery.of(context).size.width;
    double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      extendBodyBehindAppBar: true,  // Makes the body extend behind the app bar
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: isUserSignedIn,
        backgroundColor: Colors.transparent,  // Makes the app bar transparent

      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn),
      body: BackgroundWrapper(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),  // Padding for better spacing
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,  // Centering content horizontally
            children: [
              // Add padding at the top to avoid overlap with the AppBar
              SizedBox(height: 100),  // Adjust this value based on your AppBar height

              // About Section
              Text(
                'About PresentSense'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white,
                  fontSize: screenWidth * 0.07,  // Adjust font size based on screen width
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,  // Ensures title is centered
              ),
              SizedBox(height: 10),
              Text(
                'PresentSense is an innovative platform designed to provide seamless presentation analysis and feedback.\n'
                'Whether you are looking to improve your presentation skills, track posture, gestures, or speech fluency, '
                'we offer a comprehensive set of tools to enhance your performance.'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: screenWidth * 0.045,  // Adjust font size for body text
                  height: 1.5,
                ),
                textAlign: TextAlign.center,  // Center the text
              ),
              SizedBox(height: 20),

              // ...existing code...
// Team Section
Text(
  'Our Team'.tr(),
  style: TextStyle(
    color: Colors.white,
    fontSize: screenWidth * 0.06,
    fontWeight: FontWeight.bold,
  ),
  textAlign: TextAlign.center,
),
SizedBox(height: 10),
Wrap(
  alignment: WrapAlignment.center,
  spacing: 20,
  runSpacing: 20,
  children: [
    Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.white,
          child: Icon(
            Icons.person,
            color: Colors.black,
            size: 40,
          ),
        ),
        SizedBox(height: 5),
        Text(
          'Mayar Adel'.tr(),
          style: TextStyle(color: Colors.white70, fontSize: screenWidth * 0.035),
        ),
      ],
    ),
    Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.white,
          child: Icon(
            Icons.person,
            color: Colors.black,
            size: 40,
          ),
        ),
        SizedBox(height: 5),
        Text(
          'Arwaa Mamdoh'.tr(),
          style: TextStyle(color: Colors.white70, fontSize: screenWidth * 0.035),
        ),
      ],
    ),
    Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.white,
          child: Icon(
            Icons.person,
            color: Colors.black,
            size: 40,
          ),
        ),
        SizedBox(height: 5),
        Text(
          'Mostafa Wael'.tr(),
          style: TextStyle(color: Colors.white70, fontSize: screenWidth * 0.035),
        ),
      ],
    ),
    Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.white,
          child: Icon(
            Icons.person,
            color: Colors.black,
            size: 40,
          ),
        ),
        SizedBox(height: 5),
        Text(
          'Ahmed Yehia'.tr(),
          style: TextStyle(color: Colors.white70, fontSize: screenWidth * 0.035),
        ),
      ],
    ),
  ],
),
SizedBox(height: 20),
// ...existing code...

              // Contact Section
              Text(
                'Contact Us'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white,
                  fontSize: screenWidth * 0.06,  // Adjust font size for the contact section title
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,  // Center the section title
              ),
              SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,  // Center the contact icons and text
                children: [
                  Icon(Icons.email, color: Colors.white, size: 30),
                  SizedBox(width: 10),
                  Text(
                    'presentsense7@gmail.com'.tr(), // Use .tr() for translation
                    style: TextStyle(color: Colors.white70, fontSize: screenWidth * 0.035),
                  ),
                ],
              ),
              SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,  // Center the contact icons and text
                children: [
                  Icon(Icons.phone, color: Colors.white, size: 30),
                  SizedBox(width: 10),
                  Text(
                    '+2 01011007759'.tr(), // Use .tr() for translation
                    style: TextStyle(color: Colors.white70, fontSize: screenWidth * 0.035),
                  ),
                ],
              ),
              SizedBox(height: 20),

              // Social Media Section
              Text(
                'Follow Us'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white,
                  fontSize: screenWidth * 0.06,  // Adjust font size for the social media section title
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,  // Center the section title
              ),
              SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,  // Center the social media icons
                children: [
                  IconButton(
                    icon: Icon(Icons.facebook, color: Colors.white, size: 30),
                    onPressed: () {
                      // Add action for Facebook link
                    },
                  ),
                  IconButton(
                    icon: FaIcon(
                      FontAwesomeIcons.twitter,
                      color: Colors.white,
                      size: 30,
                    ),
                    onPressed: () {
                      // Add action for Twitter link
                    },
                  ),
                  IconButton(
                    icon: FaIcon(
                      FontAwesomeIcons.instagram,
                      color: Colors.white,
                      size: 30,
                    ),
                    onPressed: () {
                      // Add action for Instagram link
                    },
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
