from flask import Flask, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app) # Allows React to communicate with Python

@app.route('/step1_collect')
def collect():
    subprocess.Popen(['python', 'collect_data.py'])
    return jsonify({"status": "Success", "message": "Data Collection Window Opened"})

@app.route('/step2_create')
def create():
    subprocess.run(['python', 'create_dataset.py'])
    return jsonify({"status": "Success", "message": "Dataset Created (.pickle generated)"})

@app.route('/step3_train')
def train():
    subprocess.run(['python', 'train_model.py'])
    return jsonify({"status": "Success", "message": "Model Training Complete"})

@app.route('/step4_inference')
def inference():
    subprocess.Popen(['python', 'inference.py'])
    return jsonify({"status": "Success", "message": "Live Inference Started"})

if __name__ == '__main__':
    print("Bridge Server running on http://localhost:5000")
    app.run(port=5000)