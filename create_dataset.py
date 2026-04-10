import os
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- INITIALIZE THE NEW TASK API ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'
data, labels = [], []

if not os.path.exists(DATA_DIR):
    print("Error: 'data' folder not found!")
else:
    print("Task API Initialized. Starting landmark extraction...")
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path): continue
            
        print(f"Processing Class {dir_}...")
        for img_path in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, img_path))
            if img is None: continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            # Detect landmarks
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                data_aux, x_, y_ = [], [], []

                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks:
                    data_aux.append(float(landmark.x - min(x_)))
                    data_aux.append(float(landmark.y - min(y_)))

                data.append(data_aux)
                labels.append(dir_)

    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"SUCCESS! Created 'data.pickle' with {len(data)} samples.")