import requests
import json
from datetime import datetime

def print_section(title, data, indent=0):
    """Helper function to print a section of feedback with proper formatting."""
    indent_str = "  " * indent
    print(f"\n{indent_str}{'=' * (len(title) + 4)}")
    print(f"{indent_str}  {title}")
    print(f"{indent_str}{'=' * (len(title) + 4)}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent_str}  {key.replace('_', ' ').title()}: {value}")
    else:
        print(f"{indent_str}  {data}")

def save_feedback_to_file(feedback_data):
    """Save the feedback to a JSON file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feedback_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    
    return filename

def analyze_presentation(video_path, user_id):
    """Analyze a presentation video and provide comprehensive feedback."""
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'user_id': str(user_id)}
            
            response = requests.post('http://127.0.0.1:5000/upload_video', files=files, data=data)
            
            if response.status_code == 200:
                feedback_data = response.json()
                
                print("\n📊 PRESENTATION ANALYSIS REPORT")
                print("=" * 40)

                # Emotion Analysis
                print_section("Emotion Analysis", {
                    "Dominant Emotion": feedback_data.get("dominant_emotion"),
                    "Eye Contact": feedback_data.get("dominant_eye_contact"),
                    "Feedback": feedback_data.get("emotion_feedback"),
                })

                # Gesture Analysis
                print_section("Gesture Analysis", {
                    "Dominant Gesture 1": feedback_data.get("dominant_gesture_1"),
                    "Meaning 1": feedback_data.get("gesture_1_meaning"),
                    "Dominant Gesture 2": feedback_data.get("dominant_gesture_2"),
                    "Meaning 2": feedback_data.get("gesture_2_meaning"),
                    "Feedback": feedback_data.get("gesture_feedback"),
                })

                # Posture Analysis
                print_section("Posture Analysis", {
                    "Dominant Posture": feedback_data.get("dominant_posture"),
                    "Meaning": feedback_data.get("posture_meaning"),
                    "Feedback": feedback_data.get("posture_feedback"),
                })

                # Speech Analysis
                print_section("Speech Analysis", {
                    "Detected Language": feedback_data.get("detected_language"),
                    "Pace": feedback_data.get("pace"),
                    "Pace Score": feedback_data.get("pace_score"),
                    "Pace Feedback": feedback_data.get("pace_feedback"),
                    "Grammar Score": feedback_data.get("grammar_score"),
                    "Grammar Feedback": feedback_data.get("grammar_feedback"),
                    "Pronunciation Score": feedback_data.get("pronunciation_score"),
                    "Pronunciation Feedback": feedback_data.get("pronunciation_feedback"),
                    "Fluency Score": feedback_data.get("fluency_score"),
                    "Fluency Feedback": feedback_data.get("fluency_feedback"),
                    "Filler Words": feedback_data.get("filler_words"),
                })

                # Overall Score
                print_section("Overall Score", {
                    "Score": feedback_data.get("overall_score")
                })

                saved_file = save_feedback_to_file(feedback_data)
                print(f"\n✅ Feedback saved to: {saved_file}")

                return feedback_data
            else:
                print("❌ Failed to upload video. Status code:", response.status_code)
                print("Error details:", response.text)
                return None
                
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

# ✅ Fixed typo here
if __name__ == "__main__":
    video_path = 'Videos/TedTalk.mp4'
    user_id = 'test_user'

    print(f"🎥 Analyzing presentation video: {video_path}")
    feedback = analyze_presentation(video_path, user_id)
    
    if feedback:
        print("\n✨ Analysis complete! Check the feedback above for detailed insights.")
