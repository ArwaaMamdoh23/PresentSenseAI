import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # For TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "" # For PyTorch

from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import logging
from collections import Counter
from deepface import DeepFace
import torchaudio
import torchaudio.transforms as T
import tensorflow_hub as hub
import mediapipe as mp
import difflib
import speech_recognition as sr
from pydub import AudioSegment
import tensorflow as tf
import requests
from argostranslate import package, translate
import whisper
import librosa
import string
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2Processor, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    pipeline
)
from supabase import create_client, Client
import boto3
from botocore.exceptions import NoCredentialsError
import uuid
from datetime import datetime
from flask_cors import CORS




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL', 'https://ohllbliwedftnyqmthze.supabase.co'),
    os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obGxibGl3ZWRmdG55cW10aHplIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM1NDIxMzAsImV4cCI6MjA1OTExODEzMH0.XW1XNf7v3-JX94-1xJNgPM70t2qvZoEClyAab85ie1o')
)

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')  # Add your AWS region
)

# Add AWS configuration
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'your-bucket-name')
AWS_CLOUDFRONT_DOMAIN = os.getenv('AWS_CLOUDFRONT_DOMAIN', 'your-cloudfront-domain')

# Initialize models and configurations
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Wav2Vec2 model for filler word detection
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=6)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model.eval()

# Load T5 model for grammar correction
t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

# Load PoseNet Model for posture detection
posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
posenet_model = hub.load(posenet_model_url)
movenet = posenet_model.signatures['serving_default']

# Load the pronunciation evaluation pipeline
pronunciation_pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

# Load Whisper model
# Load Whisper model (force CPU)
whisper_model = whisper.load_model("base", device="cpu")




# Define mappings and counters
gesture_to_body_language = {
    "Open Palm": "Honesty",
    "Closed Fist": "Determination",
    "Pointing Finger": "Confidence",
    "Thumbs Up": "Encouragement",
    "Thumbs Down": "Criticism",
    "Victory Sign": "Peace",
    "OK Sign": "Agreement",
    "Rock Sign": "Excitement",
    "Call Me": "Friendly"
}

posture_meanings = {
    "Head Up": "Confidence",
    "Slouching": "Lack of confidence",
    "Leaning Forward": "Interest",
    "Head Down": "Lack of confidence",
    "Leaning Back": "Relaxation",
    "Arms on Hips": "Confidence",
    "Crossed Arms": "Defensive",
    "Hands in Pockets": "Casual"
}

# Initialize counters
posture_counter = {posture: 0 for posture in posture_meanings.keys()}
gesture_counter = {gesture: 0 for gesture in gesture_to_body_language.keys()}

def extract_audio_from_video(video_file_path):
    """Extract audio from video file and save as WAV."""
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    audio_file_path = os.path.join(os.path.dirname(video_file_path), f"{base_name}.wav")
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(audio_file_path)
    return audio_file_path

def predict_emotions(image):
    """Predict emotions from image using DeepFace."""
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        return "Unknown"

def detect_eye_contact(image):
    """Detect eye contact in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    return "Eye Contact" if len(eyes) > 0 else "No Eye Contact"

def refine_emotion_prediction(emotion, eye_contact):
    """Refine emotion prediction based on eye contact."""
    if emotion == "neutral":
        return "attentive" if eye_contact == "Eye Contact" else "indifferent"
    if emotion == "angry":
        return "intense" if eye_contact == "Eye Contact" else "defensive"
    if emotion == "fear":
        return "nervous" if eye_contact == "Eye Contact" else "distrust"
    if emotion == "happy":
        return "joyful" if eye_contact == "Eye Contact" else "content"
    if emotion == "sad":
        return "vulnerable" if eye_contact == "Eye Contact" else "isolated"
    if emotion == "surprise":
        return "alert" if eye_contact == "Eye Contact" else "disoriented"
    return emotion

def classify_hand_gesture(hand_landmarks):
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
    THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP = 3, 7, 11, 15, 19

    def is_finger_extended(tip, dip):
        return landmarks[tip][1] < landmarks[dip][1]

    thumb_extended = landmarks[THUMB_TIP][1] < landmarks[THUMB_IP][1]
    index_extended = is_finger_extended(INDEX_TIP, INDEX_DIP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_DIP)
    ring_extended = is_finger_extended(RING_TIP, RING_DIP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_DIP)

    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Open Palm"
    if not any([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Closed Fist"
    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing Finger"
    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up"
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down"
    if not thumb_extended and all([index_extended, middle_extended]) and not any([ring_extended, pinky_extended]):
        return "Victory Sign"
    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign"
    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign"
    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me"
    return "Unknown Gesture"


def get_posture_feedback(posture):
    """Get feedback based on detected posture."""
    feedback = {
        "Head Up": "Good posture! Keep your head up to show confidence.",
        "Slouching": "Try to stand up straight. Slouching can make you appear less confident.",
        "Leaning Forward": "Your leaning forward shows interest, but be careful not to invade personal space.",
        "Head Down": "Try to keep your head up to maintain eye contact and show confidence.",
        "Leaning Back": "Leaning back can appear relaxed, but make sure you're still engaged with your audience.",
        "Arms on Hips": "This posture shows confidence, but be mindful of appearing too aggressive.",
        "Crossed Arms": "Crossed arms can appear defensive. Try to keep your arms open to appear more approachable.",
        "Hands in Pockets": "Hands in pockets can appear casual, but might make you look less professional.",
        "Unknown Posture": "Try to maintain a balanced and confident posture."
    }
    return feedback.get(posture, "Maintain a balanced and confident posture.")

def get_gesture_feedback(gesture):
    """Get feedback based on detected gesture."""
    feedback = {
        "Open Palm": "Your open palm gestures show honesty and openness.",
        "Closed Fist": "Your closed fist shows determination, but be careful not to appear aggressive.",
        "Pointing Finger": "Pointing can be effective for emphasis, but use it sparingly to avoid appearing accusatory.",
        "Thumbs Up": "Thumbs up shows encouragement and positivity.",
        "Thumbs Down": "Be careful with thumbs down as it can appear negative.",
        "Victory Sign": "Victory sign shows enthusiasm, but use it appropriately for the context.",
        "OK Sign": "OK sign shows agreement, but be aware of cultural differences in interpretation.",
        "Rock Sign": "Rock sign shows excitement, but make sure it's appropriate for your audience.",
        "Call Me": "Call me gesture can be friendly, but ensure it's appropriate for the setting.",
        "Unknown Gesture": "Try to use clear and purposeful hand gestures to enhance your message."
    }
    return feedback.get(gesture, "Use clear and purposeful hand gestures to enhance your message.")

def get_emotion_feedback(refined_emotion, eye_contact):
    """Get feedback based on emotion and eye contact."""
    feedback = {
        "attentive": "You appear attentive. Keep your facial expressions engaging.",
        "indifferent": "Try to add more warmth to your expression for a more approachable look.",
        "intense": "Your intensity shows determination, but be mindful not to appear hostile.",
        "defensive": "Lack of eye contact with anger may appear defensive. Try to calm down and engage more openly.",
        "nervous": "Nervousness is noticeable. Maintain steady eye contact to show confidence.",
        "distrust": "Avoid avoiding eye contact; it can make you appear untrustworthy. Try to relax and engage more.",
        "joyful": "You're radiating joy! Keep the positivity and maintain eye contact for a more connected look.",
        "content": "You're happy, but make sure to engage with your audience by maintaining eye contact.",
        "vulnerable": "You seem vulnerable. Try smiling to lighten the mood if you're comfortable.",
        "isolated": "Lack of eye contact combined with sadness may appear disengaged. Try to make eye contact for a stronger presence.",
        "alert": "You seem alert! Maintain eye contact to help convey your surprise more clearly.",
        "disoriented": "You're surprised but seem disconnected. Try focusing and engaging with your audience."
    }
    return refined_emotion, feedback.get(refined_emotion, "Try to maintain a balanced expression to convey clarity.")

def get_grammar_feedback(score):
    """Get feedback based on grammar score."""
    if score >= 90:
        return "Excellent grammar! Your speech is very well-structured."
    elif score >= 80:
        return "Good grammar! There are minor improvements that could be made."
    elif score >= 70:
        return "Fair grammar. Consider reviewing your sentence structure."
    else:
        return "Your grammar needs improvement. Consider practicing more complex sentence structures."

def pronunciation_feedback(score):
    """Get feedback based on pronunciation score."""
    if score >= 90:
        return "Excellent pronunciation! Your speech is very clear."
    elif score >= 80:
        return "Good pronunciation! There are minor improvements that could be made."
    elif score >= 70:
        return "Fair pronunciation. Consider practicing difficult words more."
    else:
        return "Your pronunciation needs improvement. Consider practicing with a pronunciation guide."

def detect_interruption(audio_chunk, previous_speech_segment, silence_threshold=0.5, pace_change_threshold=50):
    """Detect interruptions in speech based on silence and pace changes."""
    silence = np.mean(np.abs(audio_chunk)) < silence_threshold
    pace_change = abs(len(audio_chunk) - len(previous_speech_segment)) > pace_change_threshold
    return silence or pace_change

def analyze_post_interruption_speech(posture, gesture, refined_emotion, eye_contact, interruptions):
    """Analyze speaker's response to interruptions and provide feedback."""
    feedback = set()  # Using a set to store feedback and eliminate duplicates
    audience_feedback = []  # Will hold the audience interaction feedback specifically

    # Check if the interruptions list is empty
    if not interruptions:
        # No interruptions, add the appropriate feedback for no interaction
        audience_feedback.append("No interaction with the audience detected.")
    else:
        # If there were interruptions, process feedback
        for interruption in interruptions:
            # Analyzing posture after interruption
            if posture == "Crossed Arms":
                feedback.add("The speaker seems defensive after the interruption.")
            elif posture == "Leaning Back":
                feedback.add("The speaker seems relaxed after the interruption.")
            elif posture == "Arms on Hips":
                feedback.add("The speaker appears confident but might be aggressive after the interruption.")
            elif posture == "Slouching":
                feedback.add("The speaker seems disengaged or insecure after the interruption.")
            elif posture == "Head Down":
                feedback.add("The speaker may feel defeated or unsure after the interruption.")
            elif posture == "Head Up":
                feedback.add("The speaker seems confident and unphased after the interruption.")
            elif posture == "Hands in Pockets":
                feedback.add("The speaker seems distant or detached after the interruption.")
            else:
                feedback.add("Unrecognized posture. Please check the input.")

            # Analyzing gestures after interruption
            if gesture == "Thumbs Up":
                feedback.add("The speaker is reassuring and positive despite the interruption.")
            elif gesture == "Thumbs Down":
                feedback.add("The speaker is likely displeased or frustrated after the interruption.")
            elif gesture == "OK Sign":
                feedback.add("The speaker may be trying to convey agreement but seems hesitant.")
            elif gesture == "Victory Sign":
                feedback.add("The speaker shows confidence and success, but might be mocking after the interruption.")
            elif gesture == "Closed Fist":
                feedback.add("The speaker seems determined but possibly frustrated after the interruption.")
            elif gesture == "Pointing Finger":
                feedback.add("The speaker may be emphasizing a point more forcefully after the interruption.")
            elif gesture == "Open Palm":
                feedback.add("The speaker is showing honesty and openness after the interruption.")
            else:
                feedback.add("Unrecognized gesture. Please check the input.")
            
           # Separate handling of refined emotions after interruption
            if refined_emotion == "attentive":
                feedback.add("You appear attentive. Keep your facial expressions engaging.")
            elif refined_emotion == "indifferent":
                feedback.add("Try to add more warmth to your expression for a more approachable look.")
            elif refined_emotion == "intense":
                feedback.add("Your intensity shows determination, but be mindful not to appear hostile.")
            elif refined_emotion == "defensive":
                feedback.add("Lack of eye contact with anger may appear defensive. Try to calm down and engage more openly.")
            elif refined_emotion == "nervous":
                feedback.add("Nervousness is noticeable. Maintain steady eye contact to show confidence.")
            elif refined_emotion == "distrust":
                feedback.add("Avoid avoiding eye contact; it can make you appear untrustworthy. Try to relax and engage more.")
            elif refined_emotion == "joyful":
                feedback.add("You're radiating joy! Keep the positivity and maintain eye contact for a more connected look.")
            elif refined_emotion == "content":
                feedback.add("You're happy, but make sure to engage with your audience by maintaining eye contact.")
            elif refined_emotion == "vulnerable":
                feedback.add("You seem vulnerable. Try smiling to lighten the mood if you're comfortable.")
            elif refined_emotion == "isolated":
                feedback.add("Lack of eye contact combined with sadness may appear disengaged. Try to make eye contact for a stronger presence.")
            elif refined_emotion == "alert":
                feedback.add("You seem alert! Maintain eye contact to help convey your surprise more clearly.")
            elif refined_emotion == "disoriented":
                feedback.add("You're surprised but seem disconnected. Try focusing and engaging with your audience.")
            else:
                feedback.add("Unknown emotion. Try to maintain a balanced expression to convey clarity.")
            
            # Analyzing eye contact after interruption
            if eye_contact == "No Eye Contact":
                feedback.add("The speaker avoids eye contact after the interruption, possibly indicating discomfort.")
            elif eye_contact == "Eye Contact":
                feedback.add("The speaker maintains eye contact, showing confidence despite the interruption.")
            else:
                feedback.add("Unrecognized eye contact. Please check the input.")

    # Combine all feedback into the final report
    if audience_feedback:
        return audience_feedback
    else:
        return list(feedback)

def run_inference(frame):
    resized_frame = cv2.resize(frame, (192, 192))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
    rgb_frame = rgb_frame / 255.0
    rgb_frame = tf.cast(rgb_frame, dtype=tf.int32)
    model_input = {"input": rgb_frame}
    return posenet_model.signatures['serving_default'](**model_input)


def extract_keypoints(results):
    # print("Shape of the output tensor:", results['output_0'].shape)
    keypoints = []
    for i in range(17):  # PoseNet detects 17 keypoints
        x = results['output_0'][0][0][i][1].numpy()
        y = results['output_0'][0][0][i][0].numpy()
        confidence = results['output_0'][0][0][i][2].numpy()
        # print(f"Keypoint {i}: x={x}, y={y}, confidence={confidence}")
        keypoints.append({
            "x": x,
            "y": y,
            "confidence": confidence
        })
    return keypoints

def classify_posture(keypoints):
    
    # Ensure keypoints are available for necessary body parts (right shoulder, hip, knee, ankle, and head)
    # try:
        shoulder_y = keypoints[5]['y']  # Right shoulder
        hip_y = keypoints[11]['y']  # Right hip
        knee_y = keypoints[13]['y']  # Right knee
        ankle_y = keypoints[15]['y']  # Right ankle
        head_y = keypoints[0]['y']  # Nose (head)

        # Check postures based on the y-coordinates of different body parts
        if abs(head_y - shoulder_y) < 0.05:
            return "Head Up"
        if shoulder_y > hip_y and knee_y > shoulder_y and ankle_y > knee_y:
            return "Slouching"
        if head_y < shoulder_y and knee_y > hip_y:
            return "Leaning Forward"
        if head_y > shoulder_y:
            return "Head Down"
        if shoulder_y < hip_y and knee_y < hip_y and ankle_y < knee_y:
            return "Leaning Back"
        if abs(keypoints[5]['x'] - keypoints[6]['x']) < 0.1 and abs(keypoints[11]['x'] - keypoints[12]['x']) < 0.1:
            return "Arms on Hips"
        if keypoints[9]['y'] < keypoints[3]['y'] and keypoints[10]['y'] < keypoints[4]['y']:
            return "Crossed Arms"
        if keypoints[9]['y'] > hip_y and keypoints[10]['y'] > hip_y:
            return "Hands in Pockets"
    
        return "Unknown Posture"

def correct_grammar(text, model, tokenizer):
    """Correct grammar in the given text using T5 model."""
    # Prepare input text
    input_text = f"grammar: {text}"
    
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate correction
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    # Decode and return corrected text
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def grammatical_score(original_text, corrected_text):
    """Calculate grammatical correctness score based on differences between original and corrected text."""
    # Convert texts to lowercase for comparison
    original = original_text.lower()
    corrected = corrected_text.lower()
    
    # Calculate similarity using difflib
    matcher = difflib.SequenceMatcher(None, original, corrected)
    similarity_ratio = matcher.ratio()
    
    # Convert similarity ratio to a score out of 100
    score = int(similarity_ratio * 100)
    
    return score

def calculate_overall_score(grammar_score, filler_score, pronunciation_score, pace, posture_score, gesture_score, eye_contact_score):
    # Define weights for each component
    weights = {
        'grammar': 0.15,        # 15% weight
        'fluency': 0.15,        # 15% weight (filler words)
        'pronunciation': 0.15,  # 15% weight
        'pace': 0.10,          # 10% weight
        'posture': 0.15,        # 15% weight
        'gesture': 0.15,        # 15% weight
        'eye_contact': 0.15     # 15% weight
    }
    
    # Normalize pace score (assuming ideal pace is between 130-160 WPM)
    if 130 <= pace <= 160:
        pace_score = 100
    elif 100 <= pace < 130 or 160 < pace <= 190:
        pace_score = 75
    else:
        pace_score = 50
    
    # Calculate weighted score
    overall_score = (
        grammar_score * weights['grammar'] +
        filler_score * weights['fluency'] +
        pronunciation_score * weights['pronunciation'] +
        pace_score * weights['pace'] +
        posture_score * weights['posture'] +
        gesture_score * weights['gesture'] +
        eye_contact_score * weights['eye_contact']
    )
    
    return round(overall_score, 2)

# ─── after you have transcription and detected_lang ─────────────────────────
def correct_with_languagetool(text: str, lang_code: str = "en") -> str:
    """
    Uses the public LanguageTool API to grammar-check text
    and returns the corrected version.
    """
    url = "https://api.languagetoolplus.com/v2/check"
    params = {"text": text, "language": lang_code, "enabledOnly": False}
    try:
        r = requests.post(url, data=params, timeout=15)
        r.raise_for_status()
        matches = r.json().get("matches", [])
    except Exception as e:
        logger.warning(f"LanguageTool request failed → returning original text: {e}")
        return text

    corrected = list(text)
    # Apply replacements starting from the end of the string so offsets stay valid
    for m in reversed(matches):
        repl = (m["replacements"][0]["value"] if m.get("replacements") else "")
        start, end = m["offset"], m["offset"] + m["length"]
        corrected[start:end] = repl
    return "".join(corrected)

# ------------------------------------------------------------------- 
# Example: Load an audio file to define waveform before transcripti

# result = whisper_model.transcribe()

# transcription = result["text"]
# detected_lang = result["language"]
# corrected_lt_sentence = correct_with_languagetool(
#     transcription, detected_lang or "en"
# )
# logger.info(f"LT-corrected sentence: {corrected_lt_sentence[:100]}…")

def upload_to_s3(file_path, bucket_name):
    """Upload a file to S3 bucket."""
    try:
        file_name = f"{uuid.uuid4()}_{os.path.basename(file_path)}"
        s3_client.upload_file(file_path, bucket_name, file_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return None

def save_feedback_to_supabase(feedback_data, user_id, file_id):
    """
    Ensures the FK (File_id) exists in Uploaded_file, then writes the Report row.
    Returns the new Report_id or None on failure.
    """
    try:
        # 1️⃣ Check if File_id already exists
        try:
            res = supabase.table("Uploaded_file").select("File_id").eq("File_id", file_id).execute()
            exists = res.data[0] if res.data else None
        except Exception as e:
            logger.warning(f"Supabase fetch failed, assuming no entry: {e}")
            exists = None

        if exists is None:
            # Insert dummy Uploaded_file row if not found
            supabase.table("Uploaded_file").insert({
                "File_id": file_id,
                "User_id": user_id,
                "File_name": f"{file_id}.mp4",
                "File_path": f"uploads/{file_id}.mp4",  
                "File_type": "video/mp4",
                "created_at": datetime.utcnow().isoformat()
            }).execute()

        # 2️⃣ Only use allowed fields
        allowed = {
            "dominant_emotion", "emotion_feedback", "dominant_eye_contact",
            "dominant_posture", "posture_meaning", "posture_feedback",
            "dominant_gesture_1", "gesture_1_meaning", "dominant_gesture_2",
            "gesture_2_meaning", "gesture_feedback", "detected_language",
            "grammar_score", "grammar_feedback", "speech_pace", "pace_score",
            "pace_feedback", "fluency_score", "filler_words", "fluency_feedback",
            "pronunciation_score", "pronunciation_feedback", "Overall_score","audience_interaction"
        }
        clean = {k: v for k, v in feedback_data.items() if k in allowed}

        # 3️⃣ Insert into Report table
        row = {
            "User_id": user_id,
            "File_id": file_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            **clean
        }

        insert_res = supabase.table("Report").insert(row).execute()
        if insert_res.data:
            return insert_res.data[0]["Report_id"]
        else:
            logger.error("Supabase insert returned no data")
            return None

    except Exception as e:
        logger.error(f"Error saving to Supabase: {e}")
        return None

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        if not video_file:
            return jsonify({"error": "Empty video file"}), 400

        os.makedirs('uploads', exist_ok=True)
        video_path = os.path.join('uploads', f"{uuid.uuid4()}.mp4")
        video_file.save(video_path)

        audio_file_path = extract_audio_from_video(video_path)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        all_emotions, all_eye_contacts, all_gestures, all_postures = [], [], [], []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                emotion = predict_emotions(frame)
                eye_contact = detect_eye_contact(frame)
                refined_emotion = refine_emotion_prediction(emotion, eye_contact)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_hands = mp_hands.process(image_rgb)

                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        gesture = classify_hand_gesture(hand_landmarks)
                        if gesture != "Unknown Gesture":
                            all_gestures.append(gesture)

                results = run_inference(frame)
                keypoints = extract_keypoints(results)
                posture = classify_posture(keypoints)
                if posture != "Unknown Posture":
                    all_postures.append(posture)

                all_emotions.append(refined_emotion)
                all_eye_contacts.append(eye_contact)

            frame_count += 1
        cap.release()

        dominant_emotion = Counter(all_emotions).most_common(1)[0][0] if all_emotions else "Unknown"
        dominant_eye_contact = Counter(all_eye_contacts).most_common(1)[0][0] if all_eye_contacts else "Unknown"
        gesture_counts = Counter(all_gestures).most_common(2)
        dominant_gesture_1 = gesture_counts[0][0] if gesture_counts else ""
        gesture_1_meaning = gesture_to_body_language.get(dominant_gesture_1, "")
        dominant_gesture_2 = gesture_counts[1][0] if len(gesture_counts) > 1 else ""
        gesture_2_meaning = gesture_to_body_language.get(dominant_gesture_2, "")
        dominant_posture = Counter(all_postures).most_common(1)[0][0] if all_postures else "Unknown"

        waveform, sr = librosa.load(audio_file_path, sr=16000)
        waveform = waveform / max(abs(waveform))

        result = whisper_model.transcribe(audio_file_path)
        transcription = result["text"]
        detected_lang = result["language"]
        corrected_lt_sentence = correct_with_languagetool(transcription, detected_lang or "en")

        audio_segment = AudioSegment.from_file(audio_file_path)
        duration = len(audio_segment) / 1000
        word_count = len(transcription.split())
        minutes = duration / 60
        pace = word_count / minutes if minutes > 0 else 0

        if 130 <= pace <= 160:
            pace_feedback = "Your pace is perfect."
        elif 100 <= pace < 130:
            pace_feedback = "You need to speed up a little bit."
        elif pace < 100:
            pace_feedback = "You are going very slow."
        elif 160 < pace <= 190:
            pace_feedback = "You need to slow down a little bit."
        else:
            pace_feedback = "You are going very fast."

        pronunciation_result = pronunciation_pipe(waveform)
        pronunciation_score = max(pronunciation_result[0]['score'] * 100, 0)
        pronunciation_feedback_text = pronunciation_feedback(pronunciation_score)
        corrected_text = correct_grammar(transcription, t5_model, tokenizer)
        grammar_score = grammatical_score(transcription, corrected_text)
        grammar_feedback_text = get_grammar_feedback(grammar_score)

        filler_counts = {"Uh": 0, "Um": 0}
        label_map = {0: "Uh", 1: "Words", 2: "Laughter", 3: "Um", 4: "Music", 5: "Breath"}

        wave, sr0 = torchaudio.load(audio_file_path)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        if sr0 != 16000:
            wave = T.Resample(sr0, 16000)(wave)
        wave = wave.squeeze(0)
        segment_samples = 2 * 16000

        for start in range(0, wave.shape[0], segment_samples):
            seg = wave[start:start + segment_samples]
            if seg.shape[0] < segment_samples:
                break
            inputs = processor(seg.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = torch.softmax(wav2vec2_model(inputs.input_values).logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            for p in preds:
                tag = label_map[int(p)]
                if tag in filler_counts:
                    filler_counts[tag] += 1

        total_chunks = max(1, wave.shape[0] // segment_samples)
        total_fillers = filler_counts["Uh"] + filler_counts["Um"]
        filler_rate = total_fillers / total_chunks
        fluency_score = max(100 - filler_rate * 100, 0)

        if fluency_score > 90:
            fluency_feedback = "Excellent — almost no fillers 🎯"
        elif fluency_score > 75:
            fluency_feedback = "Good — minor filler use 😊"
        elif fluency_score > 50:
            fluency_feedback = "Noticeable fillers; try pausing 🤮"
        else:
            fluency_feedback = "High filler use — practice pacing 🚀"

        logger.info(f"Filler counts: {filler_counts}  → fluency {fluency_score:.1f}")

        interruptions = []
        previous_speech_segment = []
        samples_per_segment = 2 * sr

        for start in range(0, len(waveform), samples_per_segment):
            end = start + samples_per_segment
            segment = waveform[start:end]
            if len(segment) < samples_per_segment:
                break
            if detect_interruption(segment, previous_speech_segment):
                interruptions.append((start, end))
            previous_speech_segment = segment

        _, emotion_feedback_text = get_emotion_feedback(dominant_emotion, dominant_eye_contact)
        gesture_feedback_text = get_gesture_feedback(dominant_gesture_1)
        posture_feedback_text = get_posture_feedback(dominant_posture)
        audience_interaction = analyze_post_interruption_speech(
            posture=dominant_posture,
            gesture=dominant_gesture_1,
            refined_emotion=dominant_emotion,
            eye_contact=dominant_eye_contact,
            interruptions=interruptions
        )

        
        # Define weights for each component
        weights = {
            'grammar': 0.15,        # 15% weight
            'fluency': 0.15,        # 15% weight (filler words)
            'pronunciation': 0.15,  # 15% weight
            'pace': 0.10,           # 10% weight
            'posture': 0.15,        # 15% weight
            'gesture': 0.15,        # 15% weight
            'eye_contact': 0.15     # 15% weight
        }

        # Calculate scores for posture, gesture, and eye contact (simple mapping: 100 if detected, else 50)
        posture_score = 100 if dominant_posture != "Unknown" else 50
        gesture_score = 100 if dominant_gesture_1 != "" else 50
        eye_contact_score = 100 if dominant_eye_contact == "Eye Contact" else 50
        filler_score = fluency_score  # Use fluency_score as filler_score

        # Normalize pace score (assuming ideal pace is between 130-160 WPM)
        if 130 <= pace <= 160:
            pace_score = 100
        elif 100 <= pace < 130 or 160 < pace <= 190:
            pace_score = 75
        else:
            pace_score = 50

        overall_score = (
            grammar_score * weights['grammar'] +
            filler_score * weights['fluency'] +
            pronunciation_score * weights['pronunciation'] +
            pace_score * weights['pace'] +
            posture_score * weights['posture'] +
            gesture_score * weights['gesture'] +
            eye_contact_score * weights['eye_contact']
        )

        feedback_data = {
            "dominant_emotion": dominant_emotion,
            "dominant_eye_contact": dominant_eye_contact,
            "emotion_feedback": emotion_feedback_text,
            "dominant_posture": dominant_posture,
            "posture_feedback": posture_feedback_text,
            "posture_meaning": posture_meanings.get(dominant_posture, ""),
            "dominant_gesture_1": dominant_gesture_1,
            "gesture_1_meaning": gesture_1_meaning,
            "dominant_gesture_2": dominant_gesture_2,
            "gesture_2_meaning": gesture_2_meaning,
            "gesture_feedback": gesture_feedback_text,
            "detected_language": detected_lang,
            "speech_pace": pace,
            "pace_score": None,
            "pace_feedback": pace_feedback,
            "pronunciation_score": pronunciation_score,
            "pronunciation_feedback": pronunciation_feedback_text,
            "grammar_score": grammar_score,
            "grammar_feedback": grammar_feedback_text,
            "fluency_score": fluency_score,
            "filler_words": filler_counts,
            "fluency_feedback": fluency_feedback,
            "Overall_score": overall_score,
            "audience_interaction": audience_interaction,
        }

        file_id = str(uuid.uuid4())
        save_feedback_to_supabase(feedback_data, user_id, file_id)

        os.remove(video_path)
        os.remove(audio_file_path)

        return jsonify(feedback_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)