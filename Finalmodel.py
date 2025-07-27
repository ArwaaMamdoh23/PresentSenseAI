import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import mediapipe as mp
from deepface import DeepFace
from collections import Counter
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, pipeline, T5ForConditionalGeneration, T5Tokenizer
import ffmpeg
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import difflib
import string
import spacy
import aifc
import os
from moviepy import VideoFileClip
import whisper
import librosa
import requests
from transformers import pipeline
from argostranslate import package, translate


# Load PoseNet Model for posture detection (PostureNet
posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
posenet_model = hub.load(posenet_model_url)

# Load face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#MediaPipe Hands for gesture recognition
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Wav2Vec2 model and processor for filler word detection
# Load Wav2Vec2 model for sequence classification

num_labels = 6
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
# wav2vec2_model.load_state_dict(torch.load("fine_tuned_wav2vec2.pth", map_location=torch.device("cpu")))
wav2vec2_model.eval()

# Pipeline for pronunciation evaluation (Using a separate variable for the pipeline)
pronunciation_pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

# Load the pre-trained T5 model for grammar correction (Use a different variable name for the T5 model)
t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")


# Gesture-to-body language mapping
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

# Define counters for gestures and postures
posture_counter = {
    "Head Up": 0,
    "Slouching": 0,
    "Leaning Forward": 0,
    "Head Down": 0,
    "Leaning Back": 0,
    "Arms on Hips": 0,
    "Hands in Pockets": 0,
    "Crossed Arms": 0
}
gesture_counter = {
    "Open Palm": 0,
    "Closed Fist": 0,
    "Pointing Finger": 0,
    "Thumbs Up": 0,
    "Thumbs Down": 0,
    "Victory Sign": 0,
    "OK Sign": 0,
    "Rock Sign": 0,
    "Call Me": 0
}

# Facial emotion recognition using DeepFace
def predict_emotions(image):
    try:
        # DeepFace analyzes the image and gets the dominant emotion
        analysis = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "Unknown"

def refine_emotion_prediction(emotion, eye_contact):
    if emotion == "neutral":
        if eye_contact == "Eye Contact":
            return "attentive"
        else:
            return "indifferent"
    
    if emotion == "angry":
        if eye_contact == "Eye Contact":
            return "intense"
        else:
            return "defensive"
    
    if emotion == "fear":
        if eye_contact == "Eye Contact":
            return "nervous"
        else:
            return "distrust"
    
    if emotion == "happy":
        if eye_contact == "Eye Contact":
            return "joyful"
        else:
            return "content"
    
    if emotion == "sad":
        if eye_contact == "Eye Contact":
            return "vulnerable"
        else:
            return "isolated"
    
    if emotion == "surprise":
        if eye_contact == "Eye Contact":
            return "alert"
        else:
            return "disoriented"
    
    return emotion


def detect_eye_contact(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    return "Eye Contact" if len(eyes) > 0 else "No Eye Contact"

# Gesture classification function
def classify_hand_gesture(hand_landmarks):
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
    THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP = 3, 7, 11, 15, 19
    THUMB_MCP = 2

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

# Function to extract keypoints from PoseNet output
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

# Updated Posture classification logic (Modified part)
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


# Run PoseNet Inference
def run_inference(frame):
    resized_frame = cv2.resize(frame, (192, 192))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
    rgb_frame = rgb_frame / 255.0
    rgb_frame = tf.cast(rgb_frame, dtype=tf.int32)
    model_input = {"input": rgb_frame}
    return posenet_model.signatures['serving_default'](**model_input)

# Main video processing loop
# cap = cv2.VideoCapture("Videos/TedDutch.mp4")
# cap = cv2.VideoCapture("Videos/lolo presentation.mp4")
cap = cv2.VideoCapture("Videos/TedTalk.mp4")
# cap = cv2.VideoCapture("Videos/TedChinese.mp4")

# Set the resolution to high values (e.g., Full HD: 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")  # Print FPS
frame_count = 0
is_paused = False
playback_speed = 1
all_emotions = []
all_eye_contacts = []

# Initialize variables for display
posture = "Unknown"
gesture = "Unknown"
refined_emotion = "Unknown"
eye_contact = "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Every 10th frame to detect emotions (optimization)
    if frame_count % 10 == 0:
        # Run inference to detect posture
        results = run_inference(frame)
        keypoints = extract_keypoints(results)
        posture = classify_posture(keypoints)

        # Draw posture keypoints
        for keypoint in keypoints:
            # print(f"Keypoint: x={keypoint['x']}, y={keypoint['y']}, confidence={keypoint['confidence']}")
            if 0 <= keypoint['x'] <= 1 and 0 <= keypoint['y'] <= 1 and keypoint['confidence'] > 0.5:
                x = int(keypoint['x'] * frame.shape[1])
                y = int(keypoint['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Ensure the posture is a valid one before incrementing
        if posture in posture_counter:
            posture_counter[posture] += 1
        else:
            print(f"Detected an invalid posture: {posture}")

        # print(f"Detected Posture: {posture}")
        gesture = "Unknown Gesture"
        # Process MediaPipe for Gesture Detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = mp_hands.process(image_rgb)

        # Draw only hand bounding boxes (no landmarks)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get bounding box for hand
                h, w, _ = frame.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Draw bounding box around hand with thicker lines
                cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (255, 0, 0), 3)
                
                gesture = classify_hand_gesture(hand_landmarks)
                if gesture != "Unknown Gesture":
                    gesture_counter[gesture] += 1
                # print(f"Detected Gesture: {gesture}")

        # Detect face and draw face box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Detect emotions and eye contact
            face_roi = frame[y:y+h, x:x+w]
            emotion = predict_emotions(face_roi)
            eye_contact = detect_eye_contact(face_roi)
            refined_emotion = refine_emotion_prediction(emotion, eye_contact)
            # print(f"Emotion: {refined_emotion}, Eye Contact: {eye_contact}")

            # Append detected emotion and eye contact to the lists
            all_emotions.append(refined_emotion)
            all_eye_contacts.append(eye_contact)

    frame_count += 1

    # Display the frame with detected information
    cv2.putText(frame, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Emotion: {refined_emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Eye Contact: {eye_contact}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print dominant emotion and eye contact if available
if all_emotions:
    dominant_emotion = Counter(all_emotions).most_common(1)[0][0]  
    print(f"Dominant Emotion: {dominant_emotion}")

if all_eye_contacts:
    dominant_eye_contact = Counter(all_eye_contacts).most_common(1)[0][0]  
    print(f"Dominant Eye Contact: {dominant_eye_contact}")

# Print posture counts with meanings
# print("\nPosture Count Summary:")
for posture, count in posture_counter.items():
    meaning = posture_meanings.get(posture, "Unknown Meaning")
    # print(f"{posture}: {count} times - Meaning: {meaning}")

# Print most repeated gestures with meaning
# print("\nMost Repeated Gestures with Meaning: ")
# print("\nGesture Summary:")
for gesture, count in gesture_counter.items():
    meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")
    # print(f"{gesture}: {count} times - Meaning: {meaning}")
    
    
def extract_audio_from_video(video_file_path):
    # Get the base name of the video file (without extension)
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    
    # Generate the audio file path by changing the extension to .wav
    audio_file_path = os.path.join(os.path.dirname(video_file_path), f"{base_name}.wav")
    
    # Extract audio from the video
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(audio_file_path)  # Save audio as .wav file
    
    return audio_file_path

# Specify your video file path
# video_file_path = "Videos/lolo presentation.mp4"
video_file_path = "Videos/TedTalk.mp4"
# video_file_path = "Videos/TedDutch.mp4"
# video_file_path = "Videos/TedChinese.mp4"


# Automatically generate the audio file path and extract the audio
audio_file_path = extract_audio_from_video(video_file_path)

# Print the generated audio file path
# print(f"Audio extracted and saved at: {audio_file_path}")


model = whisper.load_model("base")

# Load audio using librosa
waveform, sr = librosa.load(audio_file_path, sr=16000)
waveform = waveform / max(abs(waveform))  # Normalize

# Transcribe using Whisper
result = model.transcribe(audio=waveform)
transcription = result["text"]
detected_lang = result["language"]
print("Transcription:", transcription, "\n")


def correct_grammar(sentence, t5_model, tokenizer, max_length=512):
    corrected_text = ""
    i = 0
    while i < len(sentence):
        # Get the next chunk of the sentence
        chunk = sentence[i:i + max_length]
        # Prepare input with prefix "grammar: " for T5 model
        input_text = "grammar: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

        # Generate corrected text
        outputs = t5_model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        corrected_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append the corrected chunk
        corrected_text += corrected_chunk + " "

        # Move to the next chunk
        i += max_length

    return corrected_text.strip()



corrected_sentence = correct_grammar(transcription, t5_model, tokenizer)


# Function to normalize text by removing punctuation and converting to lowercase
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

corrected_sentence = normalize_text(corrected_sentence)
transcription = normalize_text(transcription)

def grammatical_score(original, corrected):
    original_normalized = normalize_text(original)
    corrected_normalized = normalize_text(corrected)

    diff = difflib.ndiff(original_normalized.split(), corrected_normalized.split())
    changes = list(diff)

    corrected_words = [word for word in changes if word.startswith('+ ')]
    removed_words = [word for word in changes if word.startswith('- ')]

    num_changes = len(corrected_words) + len(removed_words)
    total_words = len(original_normalized.split())

    error_ratio = num_changes / total_words if total_words > 0 else 0

    if error_ratio == 0:
        return 10
    elif error_ratio < 0.05:
        return 9
    elif error_ratio < 0.1:
        return 8
    elif error_ratio < 0.2:
        return 6
    else:
        return 4
grammar_score = grammatical_score(transcription, corrected_sentence)

def get_grammar_feedback(score):
    if score == 10:
        return "Perfect grammar! Keep up the great work! 🎯"
    elif score >= 9:
        return "Very good grammar! A little improvement could make it perfect. 👍"
    elif score >= 8:
        return "Good grammar! There are a few minor mistakes. Keep improving. 😊"
    elif score >= 6:
        return "Grammar needs improvement. Review some rules and practice more. 📝"
    else:
        return "Poor grammar. It might help to practice more and focus on sentence structure. 🚀"


def calculate_speech_pace(audio_file):
    # Use pydub to get the duration of the audio
    audio_segment = AudioSegment.from_file(audio_file)
    duration = len(audio_segment) / 1000  # seconds
    # Load audio waveform using librosa (no ffmpeg)
    waveform, sr = librosa.load(audio_file, sr=16000)
    waveform = waveform / max(abs(waveform))  # Normalize
    # Transcribe with Whisper using waveform
    result = model.transcribe(audio=waveform)
    transcription = result["text"]
    # Calculate pace
    word_count = len(transcription.split())
    minutes = duration / 60
    pace = word_count / minutes if minutes > 0 else 0
    # Feedback
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
    return pace, transcription, pace_feedback
# Check if result is None before unpacking
pace_result = calculate_speech_pace(audio_file_path)

if pace_result is None:
    print("Error: Could not process the audio.")
else:
    pace, transcribed_text, pace_feedback = pace_result
    if transcribed_text:
        # print("Transcription:", transcribed_text, "\n")
        print(f"Your grammatical score was: {grammatical_score(transcribed_text, corrected_sentence)}/10")
        print("Speech Pace (WPM):", pace)
        print("Feedback:", pace_feedback)
    else:
        print("No transcription available.")

def correct_with_languagetool(text, lang_code):
        url = "https://api.languagetoolplus.com/v2/check"
        params = {
            "text": text,
            "language": lang_code,
            "enabledOnly": False
        }
        response = requests.post(url, data=params)
        matches = response.json().get("matches", [])
        corrected_text = text
        for match in reversed(matches):
            offset = match["offset"]
            length = match["length"]
            replacement = match["replacements"][0]["value"] if match["replacements"] else ""
            corrected_text = corrected_text[:offset] + replacement + corrected_text[offset+length:]
        return corrected_text
    # Use it on the Whisper transcription
corrected_lt_sentence = correct_with_languagetool(transcription, detected_lang)
print("Corrected (via LanguageTool):", corrected_lt_sentence)

# Load and resample the audio file for filler word detection
waveform, sample_rate = torchaudio.load(audio_file_path)
waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# Define chunking parameters
segment_duration = 2
sample_rate = 16000
samples_per_segment = segment_duration * sample_rate

total_duration = waveform.shape[1] / sample_rate
total_segments = total_duration // segment_duration

filler_counts = {"Uh": 0, "Um": 0}
label_map = {0: "Uh", 1: "Words", 2: "Laughter", 3: "Um", 4: "Music", 5: "Breath"}

# Process audio in 2-second chunks for filler word detection
for start in range(0, waveform.shape[1], samples_per_segment):
    end = start + samples_per_segment
    segment = waveform[:, start:end]

    if segment.shape[1] < samples_per_segment:
        break

    inputs = processor(segment.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = wav2vec2_model(inputs.input_values).logits
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    for pred in predictions:
        label = label_map[pred.item()]
        if label in filler_counts:
            filler_counts[label] += 1

# Compute filler rate
total_fillers = filler_counts["Uh"] + filler_counts["Um"]
filler_rate = total_fillers / total_segments if total_segments > 0 else 0

filler_score = max(100 - (filler_rate * 100), 0)

filler_feedback = ""
if filler_score > 90:
    filler_feedback = "Excellent! You used very few filler words. Keep it up! 🎯"
elif filler_score > 75:
    filler_feedback = "Good job! You have some fillers, but it's not distracting. Try to be more mindful. 😊"
elif filler_score > 50:
    filler_feedback = "You're using filler words often. Consider pausing instead of saying 'Uh' or 'Um'. 🧐"
else:
    filler_feedback = "High use of filler words detected! Try slowing down and structuring your thoughts before speaking. 🚀"


# Print results
print(f"\n🎤 Speech Fluency Score: {filler_score:.2f}/100")
print(f"📝 Feedback: {filler_feedback}")
print(f"📊 Filler Word Breakdown: {filler_counts}")


# Load the audio file (use audio_file_path, not audio_file)
waveform, sample_rate = torchaudio.load(audio_file_path, format="wav")

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed (Wav2Vec2 expects 16kHz audio)
if sample_rate != 16000:
    resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resample(waveform)

# Normalize the waveform
waveform = (waveform - waveform.mean()) / waveform.std()

# Convert tensor to numpy array (if needed for the pipeline)
waveform = waveform.squeeze(0).numpy()

# Run the pronunciation evaluation model
result = pronunciation_pipe(waveform) 

# Define score mapping
score_mapping = {
    "advanced": 100,
    "proficient": 75,
    "intermediate": 50,
    "beginer": 25  # Note: 'beginer' is misspelled in the model output, so keeping it as is
}

# Compute final score based on the model output
final_pronunciation_score = sum(score_mapping[item["label"]] * item["score"] for item in result)

# Load and resample the audio file for filler word detection
waveform, sample_rate = torchaudio.load(audio_file_path)
print(f"Waveform shape: {waveform.shape}")  # Check the shape of waveform

# If stereo (more than 1 channel), convert it to mono by averaging the channels
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed (Wav2Vec2 expects 16kHz audio)
if sample_rate != 16000:
    resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resample(waveform)

# Normalize the waveform
waveform = (waveform - waveform.mean()) / waveform.std()

# Convert tensor to numpy array (if needed for the pipeline)
waveform = waveform.squeeze(0).numpy()

# Define chunking parameters
segment_duration = 2  # in seconds
sample_rate = 16000  # Resampled sample rate
samples_per_segment = segment_duration * sample_rate


def pronunciation_feedback(score):
    if score >= 90:
        return "Excellent pronunciation! Your articulation is clear and precise. 🎯"
    elif score >= 75:
        return "Good pronunciation. A few improvements can make your speech clearer. 😊"
    elif score >= 50:
        return "Pronunciation could be better. Focus on articulation and clarity. 🧐"
    else:
        return "Needs improvement in pronunciation. Practice speaking more clearly. 🚀"
    
pronunciation_feedback = pronunciation_feedback(final_pronunciation_score)   

def detect_interruption(audio_chunk, previous_speech_segment, silence_threshold=0.5, pace_change_threshold=50):
    silence = np.mean(np.abs(audio_chunk)) < silence_threshold  # Silence detection
    pace_change = abs(len(audio_chunk) - len(previous_speech_segment)) > pace_change_threshold  # Pace change detection

    if silence or pace_change:
        return True
    return False

# Process the audio chunk and detect interruptions
interruptions = []
previous_speech_segment = []  # Stores the previous chunk of the speaker's speech

# Load and resample the audio file for filler word detection
waveform, sample_rate = torchaudio.load(audio_file_path)
print(f"Waveform shape: {waveform.shape}")  # Check the shape of waveform

# If stereo (more than 1 channel), convert it to mono by averaging the channels
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed (Wav2Vec2 expects 16kHz audio)
if sample_rate != 16000:
    resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resample(waveform)

# Normalize the waveform
waveform = (waveform - waveform.mean()) / waveform.std()

# Convert tensor to numpy array (if needed for the pipeline)
waveform = waveform.squeeze(0).numpy()

# Define chunking parameters
segment_duration = 2  # in seconds
sample_rate = 16000  # Resampled sample rate
samples_per_segment = segment_duration * sample_rate

# Process the audio in chunks
interruptions = []
previous_speech_segment = []  # Stores the previous chunk of the speaker's speech

for start in range(0, waveform.shape[0], samples_per_segment):
    end = start + samples_per_segment
    segment = waveform[start:end]  # No need to slice by channels

    if segment.shape[0] < samples_per_segment:  # Handle remaining chunk size
        break

    # Process the segment (your existing code for interruption detection)
    interruption_detected = detect_interruption(segment, previous_speech_segment)
    if interruption_detected:
        interruptions.append((start, end))  # Store the time interval when interruption occurred

    previous_speech_segment = segment  # Update previous segment

# The interruptions list will now contain the times of the interruptions

# After detecting interruptions, analyze the speaker's response
def analyze_post_interruption_speech(posture, gesture, emotion, eye_contact, interruptions):
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
            
            # Analyzing emotion after interruption
            if emotion == "angry":
                feedback.add("The speaker seems angry after the interruption.")
            elif emotion == "happy":
                feedback.add("The speaker seems joyful and unaffected by the interruption.")
            elif emotion == "sad":
                feedback.add("The speaker may feel discouraged or upset after the interruption.")
            elif emotion == "fear":
                feedback.add("The speaker seems nervous or anxious after the interruption.")
            elif emotion == "surprise":
                feedback.add("The speaker is taken aback by the interruption and looks surprised.")
            elif emotion == "neutral" or emotion not in ["angry", "happy", "sad", "fear", "surprise"]:
                feedback.add("The speaker maintains a neutral expression, unaffected by the interruption.")
            elif emotion == "attentive":
                feedback.add("You appear attentive. Keep your facial expressions engaging.")
            else:
                feedback.add("Unrecognized emotion. Please check the input.")
            
            # Analyzing eye contact after interruption
            if eye_contact == "No Eye Contact":
                feedback.add("The speaker avoids eye contact after the interruption, possibly indicating discomfort.")
            elif eye_contact == "Eye Contact":
                feedback.add("The speaker maintains eye contact, showing confidence despite the interruption.")
            else:
                feedback.add("Unrecognized eye contact. Please check the input.")

    # Combine all feedback into the final report, ensuring we don't add both no interaction and interruption feedback
    if audience_feedback:
        # If there were no interruptions, return only the no-interaction message
        return audience_feedback
    else:
        # If there were interruptions, return the relevant feedback
        return list(feedback)


# Testing with no interruptions
result_no_interruption = analyze_post_interruption_speech("Head Up", "Thumbs Up", "attentive", "Eye Contact", [])
print("Audience Interaction Feedback ---")
print(result_no_interruption)

# Testing with valid interruptions
result_with_interruptions = analyze_post_interruption_speech("Leaning Back", "Thumbs Down", "fear", "No Eye Contact", ["interruption1"])
print("Audience Interaction Feedback ---")
print(result_with_interruptions)


# Get response feedback
response_feedback = analyze_post_interruption_speech(posture, gesture, refined_emotion, eye_contact, interruptions)



 
# Add posture improvement tips to feedback
def get_posture_feedback(posture):
    if posture == "Head Up":
        return "Great posture! This shows confidence. Keep it up!"
    elif posture == "Slouching":
        return "Try to sit or stand up straighter to appear more confident."
    elif posture == "Leaning Forward":
        return "This indicates interest, but be mindful not to overdo it as it may appear overly eager."
    elif posture == "Head Down":
        return "You may be feeling insecure. Try to lift your head and maintain eye contact for more presence."
    elif posture == "Leaning Back":
        return "Relaxed, but make sure it doesn't look disengaged. Stay attentive."
    elif posture == "Arms on Hips":
        return "A confident posture, but ensure it's not overly aggressive."
    elif posture == "Crossed Arms":
        return "Crossed arms can indicate defensiveness. Consider keeping your arms open for a more approachable look."
    elif posture == "Hands in Pockets":
        return "This is a casual posture, but might convey a lack of engagement. Try to relax your arms by your sides."
    else:
        return "Unknown posture. Make sure to maintain an open and relaxed stance."

# Add gesture improvement tips to feedback
def get_gesture_feedback(gesture):
    if gesture == "Open Palm":
        return "Great gesture! This can indicate honesty and openness."
    elif gesture == "Closed Fist":
        return "This gesture conveys determination but can be seen as aggressive. Consider using more open gestures."
    elif gesture == "Pointing Finger":
        return "Pointing can appear accusatory. Use it sparingly and avoid directing it at people."
    elif gesture == "Thumbs Up":
        return "A positive gesture. Use it to express approval or encouragement."
    elif gesture == "Thumbs Down":
        return "This can be interpreted negatively. Avoid using this in most social settings."
    elif gesture == "Victory Sign":
        return "The peace gesture shows positivity and can symbolize success or peace."
    elif gesture == "OK Sign":
        return "A good gesture for agreement. Just be cautious about the context, as it may have different meanings in other cultures."
    elif gesture == "Rock Sign":
        return "A fun, energetic gesture that expresses excitement. Keep it lighthearted!"
    elif gesture == "Call Me":
        return "A playful gesture that indicates friendliness. Use it in informal settings."
    else:
        return "Unknown gesture. Be mindful of the context and cultural interpretations of different gestures."

# Add emotion and eye contact improvement tips
def get_emotion_feedback(refined_emotion, eye_contact):
    # Use the refined emotion here instead of raw emotion
    if refined_emotion == "attentive":
        feedback = "You appear attentive. Keep your facial expressions engaging."
    elif refined_emotion == "indifferent":
        feedback = "Try to add more warmth to your expression for a more approachable look."
    elif refined_emotion == "intense":
        feedback = "Your intensity shows determination, but be mindful not to appear hostile."
    elif refined_emotion == "defensive":
        feedback = "Lack of eye contact with anger may appear defensive. Try to calm down and engage more openly."
    elif refined_emotion == "nervous":
        feedback = "Nervousness is noticeable. Maintain steady eye contact to show confidence."
    elif refined_emotion == "distrust":
        feedback = "Avoid avoiding eye contact; it can make you appear untrustworthy. Try to relax and engage more."
    elif refined_emotion == "joyful":
        feedback = "You're radiating joy! Keep the positivity and maintain eye contact for a more connected look."
    elif refined_emotion == "content":
        feedback = "You're happy, but make sure to engage with your audience by maintaining eye contact."
    elif refined_emotion == "vulnerable":
        feedback = "You seem vulnerable. Try smiling to lighten the mood if you're comfortable."
    elif refined_emotion == "isolated":
        feedback = "Lack of eye contact combined with sadness may appear disengaged. Try to make eye contact for a stronger presence."
    elif refined_emotion == "alert":
        feedback = "You seem alert! Maintain eye contact to help convey your surprise more clearly."
    elif refined_emotion == "disoriented":
        feedback = "You're surprised but seem disconnected. Try focusing and engaging with your audience."
    else:
        feedback = "Unknown emotion. Try to maintain a balanced expression to convey clarity."
    
    return refined_emotion, feedback

# Get detailed feedback for each posture, gesture, and emotion detected
# Get detailed feedback for each posture, gesture, and emotion detected
posture_feedback = get_posture_feedback(posture)
gesture_feedback = get_gesture_feedback(gesture)
refined_emotion, emotion_feedback = get_emotion_feedback(dominant_emotion, dominant_eye_contact)


result = calculate_speech_pace(audio_file_path)

# Check if the result is None (in case speech recognition failed)
if result is None:
    print("Error: Could not process the audio.")
    # Handle the case where there is no speech pace feedback
    pace = None
    transcription = None
    pace_feedback = None
else:
    pace, transcription, pace_feedback = result
## Provide feedback report
combined_feedback_report = []

# Add dominant emotion and eye contact
combined_feedback_report.append(f"Dominant Emotion: {dominant_emotion}")
combined_feedback_report.append(f"Dominant Eye Contact: {dominant_eye_contact}")
# Add the emotion feedback
combined_feedback_report.append(f"Emotion Feedback: {emotion_feedback}")

print(f"Refined Emotion: {refined_emotion}")
print(f"Emotion Feedback: {emotion_feedback}")

dominant_posture, posture_count = Counter(posture_counter).most_common(1)[0]
posture_feedback = get_posture_feedback(dominant_posture)
# Add posture analysis
# combined_feedback_report.append("\n--- Posture Analysis ---")
# if dominant_posture != "Unknown":
#     meaning = posture_meanings.get(dominant_posture, "Unknown Meaning")
#     combined_feedback_report.append(f"Dominant Posture: {dominant_posture} - Meaning: {meaning}")
# combined_feedback_report.append(f"Posture Feedback: {posture_feedback}")



dominant_gesture, gesture_count = Counter(gesture_counter).most_common(1)[0]
# gesture_feedback = get_gesture_feedback(dominant_gesture)
# # Add gesture analysis (top 2 most frequent)
# combined_feedback_report.append("\n--- Gesture Analysis ---")
# for gesture, count in Counter(gesture_counter).most_common(2):
#     if count > 0:
#         meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")
#         combined_feedback_report.append(f"Dominant Gesture: {gesture} - Meaning: {meaning}")

# combined_feedback_report.append(f"Gesture Feedback: {gesture_feedback}")


# Add speech analysis feedback (grammar, pace, fluency, pronunciation)
# combined_feedback_report.append("\n--- Speech Analysis ---")
# combined_feedback_report.append(f"Detected Language: {detected_lang}")

# Initialize all variables needed for the feedback report
grammar_percentage = (grammar_score / 10) * 100 if grammar_score is not None else 0
filler_percentage = filler_score if filler_score is not None else 0
pronunciation_percentage = final_pronunciation_score if final_pronunciation_score is not None else 0

# Calculate pace score (ideal range: 130-160 WPM)
if pace is not None:
    if 130 <= pace <= 160:
        pace_percentage = 100
    elif 100 <= pace < 130 or 160 < pace <= 190:
        pace_percentage = 75
    else:
        pace_percentage = 50
else:
    pace_percentage = 0

# Calculate posture score based on confidence-indicating postures
confident_postures = ["Head Up", "Arms on Hips", "Leaning Forward"]
total_postures = sum(posture_counter.values()) if posture_counter else 0
if total_postures > 0:
    confident_posture_count = sum(posture_counter[posture] for posture in confident_postures if posture in posture_counter)
    posture_percentage = (confident_posture_count / total_postures) * 100
else:
    posture_percentage = 0

# Calculate gesture score based on positive gestures
positive_gestures = ["Open Palm", "Thumbs Up", "Victory Sign", "OK Sign"]
total_gestures = sum(gesture_counter.values()) if gesture_counter else 0
if total_gestures > 0:
    positive_gesture_count = sum(gesture_counter[gesture] for gesture in positive_gestures if gesture in gesture_counter)
    gesture_percentage = (positive_gesture_count / total_gestures) * 100
else:
    gesture_percentage = 0

# Calculate emotion and eye contact score
positive_emotions = ["happy", "attentive", "joyful", "content"]
emotion_score = 100 if refined_emotion in positive_emotions else 50
eye_contact_score = 100 if eye_contact == "Eye Contact" else 50

# Initialize feedback variables
posture_feedback = get_posture_feedback(posture) if posture else "No posture data available"
gesture_feedback = get_gesture_feedback(gesture) if gesture else "No gesture data available"
emotion_feedback = get_emotion_feedback(refined_emotion, eye_contact)[1] if refined_emotion else "No emotion data available"

# Add speech analysis feedback (grammar, pace, fluency, pronunciation)
combined_feedback_report.append("\n--- Speech Analysis ---")
combined_feedback_report.append(f"Detected Language: {detected_lang if detected_lang else 'Unknown'}")

# Grammar Section
combined_feedback_report.append("\n --- Grammar Analysis ---")
combined_feedback_report.append(f"Grammar Score: {grammar_score}/10 ({grammar_percentage}%)")
# combined_feedback_report.append(f"Original Text: {transcription if transcription else 'No transcription available'}")
# combined_feedback_report.append(f"Corrected Text (T5): {corrected_sentence if corrected_sentence else 'No correction available'}")
# combined_feedback_report.append(f"Corrected Text (LanguageTool): {corrected_lt_sentence if corrected_lt_sentence else 'No correction available'}")
combined_feedback_report.append(f"Grammar Feedback: {get_grammar_feedback(grammar_score) if grammar_score is not None else 'No grammar analysis available'}")

# Pace Section
combined_feedback_report.append("\n --- Speech Pace Analysis ---")
combined_feedback_report.append(f"Speech Pace: {pace if pace is not None else 'N/A'} WPM")
combined_feedback_report.append(f"Pace Score: {pace_percentage}%")
combined_feedback_report.append(f"Pace Feedback: {pace_feedback if pace_feedback else 'No pace feedback available'}")

# Fluency Section
combined_feedback_report.append("\n --- Fluency Analysis ---")
combined_feedback_report.append(f"Fluency Score: {filler_score if filler_score is not None else 'N/A'}/100")
combined_feedback_report.append(f"Filler Word Breakdown: {filler_counts if filler_counts else 'No filler word data available'}")
combined_feedback_report.append(f"Fluency Feedback: {filler_feedback if filler_feedback else 'No fluency feedback available'}")

# Pronunciation Section
combined_feedback_report.append("\n --- Pronunciation Analysis ---")
combined_feedback_report.append(f"Pronunciation Score: {final_pronunciation_score if final_pronunciation_score is not None else 'N/A'}/100")
combined_feedback_report.append(f"Pronunciation Feedback: {pronunciation_feedback if pronunciation_feedback else 'No pronunciation feedback available'}")

# Body Language Analysis
combined_feedback_report.append("\n--- Body Language Analysis ---")

combined_feedback_report.append("\n--- Posture Analysis ---")
if dominant_posture != "Unknown":
    meaning = posture_meanings.get(dominant_posture, "Unknown Meaning")
    combined_feedback_report.append(f"Dominant Posture: {dominant_posture} - Meaning: {meaning}")
combined_feedback_report.append(f"Posture Feedback: {posture_feedback}")


gesture_feedback = get_gesture_feedback(dominant_gesture)
# Add gesture analysis (top 2 most frequent)
combined_feedback_report.append("\n--- Gesture Analysis ---")
for gesture, count in Counter(gesture_counter).most_common(2):
    if count > 0:
        meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")
        combined_feedback_report.append(f"Dominant Gesture: {gesture} - Meaning: {meaning}")

combined_feedback_report.append(f"Gesture Feedback: {gesture_feedback}")


# # Posture Section
# combined_feedback_report.append("\n --- Posture Analysis ---")
# if posture_counter:
#     for posture, count in Counter(posture_counter).most_common():
#         if count > 0:
#             meaning = posture_meanings.get(posture, "Unknown Meaning")
#             combined_feedback_report.append(f"Posture: {posture} - Count: {count} - Meaning: {meaning}")
# else:
#     combined_feedback_report.append("No posture data available")
# combined_feedback_report.append(f"Posture Score: {posture_percentage}%")
# combined_feedback_report.append(f"Posture Feedback: {posture_feedback}")

# # Gesture Section
# combined_feedback_report.append("\n --- Gesture Analysis ---")
# if gesture_counter:
#     for gesture, count in Counter(gesture_counter).most_common():
#         if count > 0:
#             meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")
#             combined_feedback_report.append(f"Gesture: {gesture} - Count: {count} - Meaning: {meaning}")
# else:
#     combined_feedback_report.append("No gesture data available")
# combined_feedback_report.append(f"Gesture Score: {gesture_percentage}%")
# combined_feedback_report.append(f"Gesture Feedback: {gesture_feedback}")

# Emotion and Eye Contact Section
combined_feedback_report.append("\n --- Emotional Analysis ---")
combined_feedback_report.append(f"Dominant Emotion: {refined_emotion if refined_emotion else 'No emotion detected'}")
combined_feedback_report.append(f"Emotion Score: {emotion_score}%")
combined_feedback_report.append(f"Eye Contact: {eye_contact if eye_contact else 'No eye contact data'}")
combined_feedback_report.append(f"Eye Contact Score: {eye_contact_score}%")
combined_feedback_report.append(f"Emotion Feedback: {emotion_feedback}")

# Audience Interaction Section
combined_feedback_report.append("\n--- Audience Interaction Analysis ---")
if response_feedback:
    for line in response_feedback:
        combined_feedback_report.append(line)
else:
    combined_feedback_report.append("No audience interaction data available")

# Calculate overall score
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

# Calculate individual scores
# Calculate posture score based on dominant posture
def calculate_posture_score(dominant_posture):
    posture_scores = {
        "Head Up": 100,
        "Arms on Hips": 90,
        "Leaning Forward": 85,
        "Leaning Back": 75,
        "Hands in Pockets": 70,
        "Crossed Arms": 65,
        "Slouching": 60,
        "Head Down": 55
    }
    return posture_scores.get(dominant_posture, 50)

# Calculate gesture score based on dominant gesture
def calculate_gesture_score(dominant_gesture):
    gesture_scores = {
        "Open Palm": 100,
        "OK Sign": 95,
        "Victory Sign": 90,
        "Thumbs Up": 85,
        "Rock Sign": 80,
        "Call Me": 75,
        "Pointing Finger": 70,
        "Thumbs Down": 65,
        "Closed Fist": 60
    }
    return gesture_scores.get(dominant_gesture, 50)

# Calculate eye contact score
def calculate_eye_contact_score(eye_contact_percentage):
    if eye_contact_percentage >= 80:
        return 100
    elif eye_contact_percentage >= 60:
        return 80
    elif eye_contact_percentage >= 40:
        return 60
    else:
        return 40

# Get dominant posture and gesture
dominant_posture = max(posture_counter.items(), key=lambda x: x[1])[0] if posture_counter else "Unknown"
dominant_gesture = max(gesture_counter.items(), key=lambda x: x[1])[0] if gesture_counter else "Unknown"

# Calculate eye contact percentage
eye_contact_percentage = (all_eye_contacts.count("Eye Contact") / len(all_eye_contacts)) * 100 if all_eye_contacts else 0

# Calculate individual scores
posture_score = calculate_posture_score(dominant_posture)
gesture_score = calculate_gesture_score(dominant_gesture)
eye_contact_score = calculate_eye_contact_score(eye_contact_percentage)

# Calculate overall score
overall_score = calculate_overall_score(
    grammar_score if grammar_score is not None else 0,
    filler_score if filler_score is not None else 0,
    final_pronunciation_score if final_pronunciation_score is not None else 0,
    pace if pace is not None else 0,
    posture_score if posture_score is not None else 0,
    gesture_score if gesture_score is not None else 0,
    eye_contact_score if eye_contact_score is not None else 0
)

# Get score interpretation
if overall_score >= 90:
    score_interpretation = "Outstanding presentation! You demonstrated excellent skills across all aspects. 🏆"
elif overall_score >= 80:
    score_interpretation = "Great presentation! You showed strong skills with minor areas for improvement. 🌟"
elif overall_score >= 70:
    score_interpretation = "Good presentation! You have a solid foundation with some areas to work on. 👍"
elif overall_score >= 60:
    score_interpretation = "Fair presentation. There are several areas that need improvement. 📝"
else:
    score_interpretation = "Needs improvement. Focus on developing your presentation skills further. 🎯"

# Overall Score Section
combined_feedback_report.append("\n--- Overall Presentation Score ---")
combined_feedback_report.append(f"Overall Score: {overall_score}/100")
combined_feedback_report.append(f"Score Interpretation: {score_interpretation}")

# Install translation model from English to the detected language
def install_translation_model(from_code, to_code):
    available_packages = package.get_available_packages()
    matching_package = next(
        (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code),
        None
    )
    if matching_package:
        download_path = matching_package.download()
        package.install_from_path(download_path)
        print(f"  translation from: {from_code} → {to_code}")
    else:
        print(f"No translation available for {from_code} → {to_code}")

# Only translate if the language isn't English
# if detected_lang != "en":
#     install_translation_model("en", detected_lang)
#     installed_languages = translate.get_installed_languages()
#     from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
#     to_lang = next((lang for lang in installed_languages if lang.code == detected_lang), None)
#     if from_lang and to_lang:
#         translator = from_lang.get_translation(to_lang)
#         translated_feedback_report = [translator.translate(line) for line in combined_feedback_report]
#         print("\n --- Translated Feedback Report ---")
#         for line in translated_feedback_report:
#             print(line)
#     else:
#         print("Translation not available. Feedback shown in English.")

def get_overall_feedback(score):
    if score >= 90:
        return "Outstanding presentation! You demonstrated excellent skills across all aspects. 🏆"
    elif score >= 80:
        return "Great presentation! You showed strong skills with minor areas for improvement. 🌟"
    elif score >= 70:
        return "Good presentation! You have a solid foundation with some areas to work on. 👍"
    elif score >= 60:
        return "Fair presentation. There are several areas that need improvement. 📝"
    else:
        return "Needs improvement. Focus on developing your presentation skills further. 🎯"

# Print the entire combined feedback report
print("\n--- Comprehensive Feedback Report ---")
for line in combined_feedback_report:
    print(line)

# Translation Logic (Add this part at the end of your script)
# Install translation model from English to the detected language
def install_translation_model(from_code, to_code):
    available_packages = package.get_available_packages()
    matching_package = next(
        (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code),
        None
    )
    if matching_package:
        download_path = matching_package.download()
        package.install_from_path(download_path)
        print(f"  translation from: {from_code} → {to_code}")
    else:
        print(f"No translation available for {from_code} → {to_code}")

# Only translate if the language isn't English
if detected_lang != "en":
    install_translation_model("en", detected_lang)
    installed_languages = translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
    to_lang = next((lang for lang in installed_languages if lang.code == detected_lang), None)
    if from_lang and to_lang:
        translator = from_lang.get_translation(to_lang)
        translated_feedback_report = [translator.translate(line) for line in combined_feedback_report]
        print("\n --- Translated Feedback Report ---")
        for line in translated_feedback_report:
            print(line)
    else:
        print("Translation not available. Feedback shown in English.")
