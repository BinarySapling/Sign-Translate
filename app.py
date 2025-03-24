import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import argparse
import time
import textwrap  # Import textwrap at the beginning
from flask import Flask, Response, render_template, jsonify
from text_to_speech import TextToSpeech

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize Flask app
app = Flask(__name__)

# Initialize the TTS engine globally
tts_engine = TextToSpeech()

# Global variables for sentence building
current_sentence = ""
last_prediction = ""
prediction_start_time = 0
prediction_duration = 2.0  # Hold sign for 2 seconds to confirm
special_commands = ["ADD", "SPACE", "DELETE", "NOTHING"]

# Load model if exists or return None
def load_model():
    if os.path.exists('sign_language_model.h5'):
        return tf.keras.models.load_model('sign_language_model.h5')
    return None

# Map prediction index to alphabet or special command
def get_prediction_text(index):
    if index < 26:
        return chr(index + 65)  # A=65, B=66, etc. in ASCII
    elif index == 26:
        return "ADD"
    elif index == 27:
        return "SPACE"
    elif index == 28:
        return "DELETE"
    elif index == 29:
        return "NOTHING"
    return "UNKNOWN"

# Process frames and make predictions
# Process frames and make predictions
def process_frame(frame, model=None):
    global current_sentence, last_prediction, prediction_start_time

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    prediction = ""
    confidence = 0

    # Get current time for tracking prediction duration
    current_time = time.time()

    # Draw hand landmarks and make prediction if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around the hand
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Extract the ROI (region of interest)
            roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess the ROI for the model
            if roi.size > 0:  # Ensure ROI is not empty
                # Extract hand landmarks for prediction
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append([landmark.x, landmark.y, landmark.z])

                # Convert to numpy array and reshape for model input
                landmarks_array = np.array(landmarks_list).flatten()  # Flattens to shape (63,)
                landmarks_input = np.expand_dims(landmarks_array, axis=0)  # Shape: (1, 63)

                # Make prediction
                if model is not None:
                    try:
                        prediction_result = model.predict(landmarks_input, verbose=0)
                        predicted_class = np.argmax(prediction_result[0])
                        confidence = prediction_result[0][predicted_class] * 100
                        prediction = get_prediction_text(predicted_class)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        prediction = ""
                        confidence = 0

            # Track prediction for adding to sentence
            if prediction == last_prediction:
                if current_time - prediction_start_time >= prediction_duration:
                    if prediction == "ADD":
                        pass
                    elif prediction == "SPACE":
                        current_sentence += " "
                        prediction_start_time = current_time
                    elif prediction == "DELETE":
                        if current_sentence:
                            current_sentence = current_sentence[:-1]
                        prediction_start_time = current_time
                    elif prediction == "NOTHING":
                        pass
                    elif prediction_start_time > 0:
                        current_sentence += prediction
                        prediction_start_time = current_time
            else:
                last_prediction = prediction
                prediction_start_time = current_time
    else:
        last_prediction = ""
        prediction_start_time = 0

    # Display prediction, confidence, and current sentence
    if prediction:
        remaining_time = max(0, prediction_duration - (current_time - prediction_start_time))
        cv2.putText(frame, f"Sign: {prediction} ({confidence:.1f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        progress = min(1.0, (current_time - prediction_start_time) / prediction_duration)
        bar_width = int(progress * 200)
        cv2.rectangle(frame, (10, 70), (210, 90), (200, 200, 200), -1)
        cv2.rectangle(frame, (10, 70), (10 + bar_width, 90), (0, 255, 0), -1)
        cv2.putText(frame, f"{remaining_time:.1f}s", (220, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "Sentence:", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    sentence_lines = textwrap.wrap(current_sentence, width=40)
    for i, line in enumerate(sentence_lines):
        cv2.putText(frame, line, (10, 160 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

# Generate frames for streaming
def generate_frames():
    camera = cv2.VideoCapture(0)
    model = load_model()

    while True:
        success, frame = camera.read()
        if not success:
            break

        processed_frame = process_frame(frame, model)

        # Convert to jpeg format for web streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

# Routes for web application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    global current_sentence
    return jsonify({'sentence': current_sentence})

@app.route('/reset_sentence')
def reset_sentence():
    global current_sentence
    tts_engine.stop()  # Stop any ongoing speech
    current_sentence = ""
    return jsonify({'result': 'success'})

@app.route('/speak_sentence')
def speak_sentence():
    global current_sentence
    if current_sentence:
        tts_engine.speak(current_sentence)
        return jsonify({'result': 'speaking', 'text': current_sentence})
    return jsonify({'result': 'nothing to speak'})

# Data collection function for training with special commands
def collect_training_data():
    print("Starting data collection for sign language training...")
    print("Including special commands: ADD, SPACE, DELETE, NOTHING")
    print("Press 's' to skip to next sign or wait for automatic collection")
    print("Press 'q' to quit")

    camera = cv2.VideoCapture(0)

    # Create directories for dataset
    os.makedirs('dataset', exist_ok=True)

    # Create directories for A-Z
    for alphabet in range(65, 91):  # A-Z
        os.makedirs(f'dataset/{chr(alphabet)}', exist_ok=True)

    # Create directories for special commands
    for command in special_commands:
        os.makedirs(f'dataset/{command}', exist_ok=True)

    # Set order of data collection
    all_signs = [chr(i) for i in range(65, 91)] + special_commands
    current_index = 0
    current_sign = all_signs[current_index]

    sample_count = 0
    max_samples = 100
    delay_counter = 0

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Process the frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Convert back to BGR for OpenCV display
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Show instructions for special commands
            if current_sign in special_commands:
                instruction = ""
                if current_sign == "ADD":
                    instruction = "Show a thumbs up gesture to add the letter"
                elif current_sign == "SPACE":
                    instruction = "Show an open palm facing forward for space"
                elif current_sign == "DELETE":
                    instruction = "Show an index finger swipe for delete"
                elif current_sign == "NOTHING":
                    instruction = "Show a closed fist for no action"

                # Add instruction text with word wrapping
                lines = textwrap.wrap(instruction, width=40)
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, 200 + i*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Save landmarks for dataset with a small delay to vary poses
                    if sample_count < max_samples and delay_counter % 3 == 0:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append([landmark.x, landmark.y, landmark.z])

                        # Save to numpy file
                        np.save(f'dataset/{current_sign}/{current_sign}_{sample_count}.npy', landmarks)
                        sample_count += 1

                    delay_counter += 1

            # Progress bar
            progress = sample_count / max_samples * 100
            bar_width = int(progress / 2)
            progress_bar = '[' + '=' * bar_width + ' ' * (50 - bar_width) + ']'

            # Display current sign and sample count
            cv2.putText(frame, f"Sign: {current_sign} ({sample_count}/{max_samples})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, progress_bar,
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Data Collection', frame)

            # Check key press
            key = cv2.waitKey(1) & 0xFF

            # If 'q' pressed, exit
            if key == ord('q'):
                break

            # If 's' pressed or enough samples collected, move to next sign
            if key == ord('s') or sample_count >= max_samples:
                current_index = (current_index + 1) % len(all_signs)
                current_sign = all_signs[current_index]
                sample_count = 0
                print(f"Moving to sign {current_sign}")

                # If we've gone through all signs
                if current_index == 0:
                    print("Data collection complete!")
                    break

    camera.release()
    cv2.destroyAllWindows()
    print("Dataset collection finished. Now you can train the model.")

def create_html_template():
    """Create the HTML template for the web application"""
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Sign Language Translator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .video-container {
            margin-top: 20px;
            position: relative;
        }
        img {
            max-width: 100%;
            border: 3px solid #333;
            border-radius: 10px;
        }
        .sentence-display {
            margin-top: 20px;
            padding: 15px;
            border: 2px solid #007bff;
            border-radius: 10px;
            font-size: 24px;
            min-height: 60px;
            background-color: #f8f9fa;
            text-align: left;
            word-wrap: break-word;
        }
        .control-buttons {
            margin-top: 15px;
        }
        .control-buttons button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 16px;
        }
        .control-buttons button:hover {
            background-color: #0056b3;
        }
        .alphabet-guide {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
            gap: 10px;
        }
        .alphabet-item {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            width: 30px;
            text-align: center;
            font-weight: bold;
        }
        .command-guide {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
            gap: 15px;
        }
        .command-item {
            background-color: #e6f7ff;
            border: 1px solid #91d5ff;
            border-radius: 5px;
            padding: 10px;
            min-width: 80px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    <script>
        // Function to update the sentence display
        function updateSentence() {
            fetch('/get_sentence')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('sentence-display').innerText = data.sentence;
                });
        }

        // Function to reset the sentence
        function resetSentence() {
            fetch('/reset_sentence')
                .then(response => response.json())
                .then(data => {
                    updateSentence();
                });
        }

        // Update the sentence every second
        window.onload = function() {
            setInterval(updateSentence, 1000);
        };

        function speakSentence() {
            fetch('/speak_sentence')
                .then(response => response.json())
                .then(data => {
                    console.log('Speaking: ', data.text);
                });
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Real-time Sign Language Translator</h1>
        <p>Hold hand signs for 2 seconds to add to your sentence</p>

        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>

        <div class="sentence-display" id="sentence-display"></div>

        <div class="control-buttons">
            <button onclick="resetSentence()">Clear Sentence</button>
            <button onclick="speakSentence()" style="background-color: #28a745;">Speak</button>
        </div>

        <h3>American Sign Language Alphabet</h3>
        <div class="alphabet-guide">
            <div class="alphabet-item">A</div>
            <div class="alphabet-item">B</div>
            <div class="alphabet-item">C</div>
            <div class="alphabet-item">D</div>
            <div class="alphabet-item">E</div>
            <div class="alphabet-item">F</div>
            <div class="alphabet-item">G</div>
            <div class="alphabet-item">H</div>
            <div class="alphabet-item">I</div>
            <div class="alphabet-item">J</div>
            <div class="alphabet-item">K</div>
            <div class="alphabet-item">L</div>
            <div class="alphabet-item">M</div>
            <div class="alphabet-item">N</div>
            <div class="alphabet-item">O</div>
            <div class="alphabet-item">P</div>
            <div class="alphabet-item">Q</div>
            <div class="alphabet-item">R</div>
            <div class="alphabet-item">S</div>
            <div class="alphabet-item">T</div>
            <div class="alphabet-item">U</div>
            <div class="alphabet-item">V</div>
            <div class="alphabet-item">W</div>
            <div class="alphabet-item">X</div>
            <div class="alphabet-item">Y</div>
            <div class="alphabet-item">Z</div>
        </div>

        <h3>Special Commands</h3>
        <div class="command-guide">
            <div class="command-item">ADD</div>
            <div class="command-item">SPACE</div>
            <div class="command-item">DELETE</div>
            <div class="command-item">NOTHING</div>
        </div>
    </div>
</body>
</html>
        ''')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Translator")
    parser.add_argument('--collect', action='store_true', help="Collect training data")
    parser.add_argument('--run', action='store_true', help="Run the web app")

    args = parser.parse_args()

    if args.collect:
        collect_training_data()
    elif args.run:
        # Create HTML template
        create_html_template()

        # Check if model exists
        if not os.path.exists('sign_language_model.h5'):
            print("Warning: Model not found. Please train the model first.")
            print("Run: python train_model.py")

        # Run Flask app
        print("Starting web application...")
        print("Access at http://127.0.0.1:5000")
        app.run(debug=True)
    else:
        # Default: show usage
        parser.print_help()
        print("\nExample usage:")
        print("  1. Collect training data: python app.py --collect")
        print("  2. Train the model:       python train_model.py")
        print("  3. Run the application:   python app.py --run")