// import 'package:flutter/material.dart';
// import '../widgets/custom_app_bar.dart';
// import '../widgets/background_wrapper.dart';

// class LoadingScreen extends StatelessWidget {
//   const LoadingScreen({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       extendBodyBehindAppBar: true,
//       appBar: CustomAppBar(
//         showSignIn: false,
//         isUserSignedIn: true,
//         backgroundColor: Colors.transparent,  // Makes the app bar transparent
//       ),
//       body: BackgroundWrapper(
//         child: Center(
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               const CircularProgressIndicator(
//                 valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
//                 strokeWidth: 4.0,
//               ),
//               const SizedBox(height: 20),
//               const Text(
//                 'Processing your data...',
//                 style: TextStyle(
//                   fontSize: 18,
//                   color: Colors.white,
//                   fontWeight: FontWeight.w500,
//                   shadows: [
//                     Shadow(
//                       blurRadius: 3.0,
//                       color: Colors.white54,
//                       offset: Offset(0, 0),
//                     ),
//                   ],
//                 ),
//               ),
//               const SizedBox(height: 10),
//               Text(
//                 'Please wait while we analyze your information',
//                 style: TextStyle(
//                   fontSize: 14,
//                   color: Colors.white.withOpacity(0.8),
//                 ),
//               ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }
