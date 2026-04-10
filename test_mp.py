import mediapipe as mp
try:
    print(f"MediaPipe Version: {mp.__version__}")
    print(f"Hands module: {mp.solutions.hands}")
    print("SUCCESS: MediaPipe is working correctly!")
except AttributeError as e:
    print(f"FAILURE: {e}")