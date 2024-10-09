from flask import Flask, render_template, request
import cv2
import torch
import numpy as np
from torchvision import transforms
from main.emotion_mapping import map_standard_to_classroom 
from main.train_model import CustomCNN
from mtcnn.mtcnn import MTCNN
import threading

app = Flask(__name__)
model = CustomCNN()  
model.load_state_dict(torch.load('student-engagement-monitoring-system-main/main/trained_model.path'))
model.eval()

mtcnn = MTCNN()

class_index_to_emotion = {
    0: 'anger',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'repression',
    6: 'sadness',
    7: 'surprise',
    8: 'tense'
}

monitoring_thread = None
monitoring_active = False

def get_available_devices(max_devices=5):
    available_devices = []
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            available_devices.append(device_id)
        cap.release()
    return available_devices

@app.route('/')
def index():
    devices = get_available_devices()  
    return render_template('index.html', devices=devices)

@app.route('/devices', methods=['GET'])
def devices():
    available_devices = get_available_devices()
    return {'devices': available_devices}

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    tensor_frame = torch.from_numpy(normalized_frame).unsqueeze(0).float()  
    return tensor_frame

def run_monitoring(device_id=0):
    global monitoring_active
    monitoring_active = True

    emotion_colors = {
        'focus': (0, 255, 0),
        'frustration': (0, 0, 255),
        'confusion': (255, 0, 0),
        'boredom': (0, 255, 255),
        'distracted': (0, 165, 255)
    }

    cap = cv2.VideoCapture(device_id)  
    cv2.namedWindow("Monitoring", cv2.WINDOW_NORMAL)

    while monitoring_active:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = mtcnn.detect_faces(frame)

        for box in boxes:
            x, y, width, height = box['box']
            input_tensor = preprocess_frame(frame).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            standard_emotion = class_index_to_emotion.get(predicted_class, 'neutral')
            classroom_emotion, intensity = map_standard_to_classroom([standard_emotion])
            color = emotion_colors.get(classroom_emotion, (255, 255, 255))

            cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), color, 2)
            cv2.putText(frame, classroom_emotion, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if cv2.getWindowProperty("Monitoring", cv2.WND_PROP_VISIBLE) >= 1:
            window_rect = cv2.getWindowImageRect("Monitoring")
            window_width, window_height = window_rect[2], window_rect[3]
            frame = cv2.resize(frame, (window_width, window_height))

        cv2.imshow("Monitoring", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start-monitoring', methods=['GET'])
def start_monitoring():
    global monitoring_thread
    # device_id = 1  # Set to a default device ID
    device_id = 1 # change to 1 for gopro
    if monitoring_thread is None or not monitoring_thread.is_alive():
        monitoring_thread = threading.Thread(target=run_monitoring, args=(device_id,))
        monitoring_thread.start()
        if device_id == 0:
            device = "WebCam"
        elif device_id == 1:
            device = "GoPro"
        return f"Monitoring started on device {device}!"
    else:
        return "Monitoring is already active."

@app.route('/stop-monitoring')
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    if monitoring_thread is not None:
        monitoring_thread.join()  # Ensure the thread finishes
    return "Monitoring stopped!"

if __name__ == '__main__':
    app.run(debug=True)
