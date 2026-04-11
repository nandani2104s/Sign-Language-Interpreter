import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# 1. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 2. Setup Data Storage
data = []
labels = []
# Change this label for each gesture you want to record (e.g., '0', '1', '2')
current_label = '0' 
required_samples = 100 # How many samples to capture automatically

cap = cv2.VideoCapture(0)

print(f"--- STARTING LIVE CAPTURE FOR LABEL: {current_label} ---")
print("Show your hand to the camera. It will capture automatically.")

while len(data) < required_samples:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton so you know it's working
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # AUTO-CAPTURE LOGIC: If a hand is found, add it to data
            data.append(data_aux)
            labels.append(current_label)

    # Visual Feedback
    cv2.putText(frame, f"PROGRESS: {len(data)}/{required_samples}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"LABEL: {current_label}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    cv2.imshow('LIVE DATA CAPTURE', frame)

    # Break if 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. Final Save
if len(data) > 0:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Successfully captured {len(data)} samples for label {current_label}!")
else:
    print("No data captured.")

cap.release()
cv2.destroyAllWindows()