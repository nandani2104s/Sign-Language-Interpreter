# Real-Time Sign Language Recognition System 🖐️
An end-to-end Machine Learning pipeline that translates hand gestures into digital commands in real-time. This project uses a **Random Forest Classifier** trained on geometric hand landmarks extracted via the **MediaPipe Task API**.

## 🚀 Project Highlights
* **Multi-Class Classification:** Recognizes 7 distinct hand gestures.
* **Optimized Inference:** Achieved **97.22% accuracy** while maintaining real-time performance (30+ FPS).
* **Robust Feature Extraction:** Uses 21-point 3D hand landmarks, making the system invariant to lighting and background noise.
* **Version Compatibility:** Custom implementation using MediaPipe Task API to ensure stability on Python 3.12/3.13.

## 📸 Demo & Gestures
The system is trained to recognize the following 7 classes:

| Class ID | Label | Gesture Description |
| :--- | :--- | :--- |
| 0 | **FIST** | All fingers folded |
| 1 | **PALM** | All fingers extended |
| 2 | **INDEX** | Only index finger pointing up |
| 3 | **THUMBS UP** | Thumb extended upward |
| 4 | **PEACE** | Index and Middle fingers extended |
| 5 | **OK** | Thumb and Index forming a circle |
| 6 | **ROCK ON** | Index and Pinky extended |

## 🛠️ Technical Stack
* **Language:** Python 3.12+
* **Framework:** MediaPipe (Task API - Vision)
* **ML Model:** Scikit-Learn (Random Forest)
* **Vision:** OpenCV
* **Data:** NumPy, Pickle

## 📂 Methodology
1. **Data Collection:** Captured 100+ frames per class using OpenCV.
2. **Feature Extraction:** Used MediaPipe to locate 21 hand landmarks ($x, y$ coordinates).
3. **Normalization:** Shifted all coordinates relative to the hand's origin $(x_i - x_{min}, y_i - y_{min})$ to ensure position-independent recognition.
4. **Training:** Trained a Random Forest ensemble to map geometric patterns to class labels.

## ⚙️ Setup & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/SignLanguageAI.git](https://github.com/yourusername/SignLanguageAI.git)
   cd SignLanguageAI