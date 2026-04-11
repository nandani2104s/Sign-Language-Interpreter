import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Please run train_model.py first.")
    exit()

# 2. Setup MediaPipe Task API with local model asset
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Initialize Webcam
cap = cv2.VideoCapture(0)

# 4. Professional Labels for your 7 Classes
labels_dict = {
    0: 'FIST / STOP', 
    1: 'PALM / HELLO', 
    2: 'INDEX / POINT', 
    3: 'THUMBS UP / YES', 
    4: 'PEACE / TWO',
    5: 'OK / PERFECT',
    6: 'ROCK ON / COOL'
}

print("Starting Real-time Inference... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Standardize frame size to resolve the NORM_RECT warning
    frame = cv2.resize(frame, (640, 480))
    H, W, _ = frame.shape
    
    # Convert to MediaPipe format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hand landmarks
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            data_aux, x_, y_ = [], [], []

            # Extract and store joint coordinates
            for landmark in hand_landmarks:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize coordinates relative to hand origin (min x, min y)
            for landmark in hand_landmarks:
                data_aux.append(float(landmark.x - min(x_)))
                data_aux.append(float(landmark.y - min(y_)))

            # Perform prediction using the Random Forest classifier
            prediction = model.predict([np.asarray(data_aux)])
            class_id = int(prediction[0])
            predicted_character = labels_dict.get(class_id, "Unknown")

            # Calculate bounding box coordinates for display
            x1, y1 = int(min(x_) * W) - 15, int(min(y_) * H) - 15
            x2, y2 = int(max(x_) * W) + 15, int(max(y_) * H) + 15
            
            # Draw UI Elements
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 255, 0), -1) # Background for text
            cv2.putText(frame, predicted_character, (x1 + 5, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the final output
    cv2.imshow('Sign Language Detector - ML Pipeline', frame)
    
    # 10ms delay allows for smoother GUI refresh
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()