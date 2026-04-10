import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 2. Setup MediaPipe Task API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Initialize Webcam
cap = cv2.VideoCapture(0)

# Define your labels based on the classes you collected (e.g., 0, 1, 2)
# You can change these to actual letters or words like {0: 'A', 1: 'B', 2: 'C'}
labels_dict = {0: '0', 1: '1', 2: '2'}

print("Starting Real-time Inference... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            data_aux, x_, y_ = [], [], []

            for landmark in hand_landmarks:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks:
                data_aux.append(float(landmark.x - min(x_)))
                data_aux.append(float(landmark.y - min(y_)))

            # Predict using the Random Forest model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw a simple bounding box and text
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()