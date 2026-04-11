import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 2. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

labels_dict = {0: 'STOP', 1: 'HELLO', 2: 'POINT', 3: 'YES', 4: 'PEACE', 5: 'OK', 6: 'COOL'}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

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

            prediction = model.predict([np.asarray(data_aux)])
            label = labels_dict[int(prediction[0])]

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (int(max(x_) * W) + 10, int(max(y_) * H) + 10), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Sign Language Detector', frame)

    # EXIT LOGIC: Press 'Q' or Click the [X] button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Sign Language Detector', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()