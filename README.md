# Sign Language Recognition System 🖐️
An AI-powered real-time Sign Language detection system built using Python, MediaPipe, and Scikit-Learn. This project extracts 21 hand landmarks to classify gestures with high precision.

## 🚀 Features
* **Real-time Detection:** 30+ FPS inference on standard CPUs.
* **High Accuracy:** Achieved **97.22% accuracy** using a Random Forest Classifier.
* **Geometric Invariance:** Uses normalized landmark coordinates, making it robust to different hand positions and camera distances.

## 📸 Demo
| Class 0 (Fist) | Class 1 (Open Palm) |
| :---: | :---: |
| ![Fist Detection](./image_535b61.jpg) | ![Palm Detection](./image_535b25.jpg) |
*(Note: Make sure to rename your images or update these paths!)*

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Computer Vision:** OpenCV, MediaPipe (Task API)
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Handling:** NumPy, Pickle

## 📂 Project Structure
* `collect_data.py`: Captures raw hand images via webcam.
* `create_dataset.py`: Extracts 21 geometric landmarks from images.
* `train_model.py`: Trains the Random Forest model on extracted features.
* `inference.py`: Real-time detection script.
* `hand_landmarker.task`: MediaPipe's pre-trained detection model.

## ⚙️ How to Run
1. **Install Dependencies:**
   ```bash
   pip install mediapipe opencv-python scikit-learn numpy<2