from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import time
import os

app = Flask(__name__)
CORS(app)

# Global variable to track the active script process
active_process = None

def run_script(script_name):
    global active_process
    
    # If a script is already running, terminate it to free the camera
    if active_process and active_process.poll() is None:
        active_process.terminate()
        try:
            active_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            active_process.kill()
        time.sleep(1) # Buffer for camera hardware reset

    try:
        # Launch the Python script as a subprocess
        active_process = subprocess.Popen(['python', script_name])
        return jsonify({"status": "Success", "message": f"Started {script_name}"})
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/step1_collect')
def collect(): return run_script('collect_data.py')

@app.route('/step2_create')
def create(): return run_script('create_dataset.py')

@app.route('/step3_train')
def train(): return run_script('train_model.py')

@app.route('/step4_inference')
def inference(): return run_script('inference.py')

if __name__ == '__main__':
    print("--- AI BRIDGE LIVE ON PORT 5000 ---")
    app.run(port=5000, debug=False)