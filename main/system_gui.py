import tkinter as tk
import threading
import cv2
import torch
import numpy as np
from torchvision import transforms
from emotion_mapping import map_standard_to_classroom 
from train_model import CustomCNN
from mtcnn.mtcnn import MTCNN

# Load the trained model
model = CustomCNN()  # Instantiate your model class
model.load_state_dict(torch.load('trained_model.path'))  # Replace 'trained_model.path' with your model path
model.eval()  # Set the model to evaluation mode

# Initialize the MTCNN detector
mtcnn = MTCNN()

# Mapping from predicted class index to standard emotion labels
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
    # 9: 'neutral'
}

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # convert to grayscale
    resized_frame = cv2.resize(gray_frame, (64, 64))        # resize images to 64 x 64
    normalized_frame = resized_frame / 255.0                # perform normalization
    # Convert to torch tensor and add channel dimension (1 channel for grayscale)
    tensor_frame = torch.from_numpy(normalized_frame).unsqueeze(0).float()  # Shape: [1, 64, 64]
        
    return tensor_frame


def start_monitoring():
    # This is where you call the function to run your system
    print("Starting monitoring...")
    
    # Define emotion colors
    emotion_colors = {
        'focus': (0, 255, 0),       # Green
        'frustration': (0, 0, 255), # Red
        'confusion': (255, 0, 0),   # Blue
        'boredom': (0, 255, 255),   # Yellow
        'distracted': (0, 165, 255) # Orange
    }

    cap = cv2.VideoCapture(0)  # Capture from webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces and get bounding boxes
        boxes = mtcnn.detect_faces(frame)

        # Draw bounding boxes and get emotions as needed
        for box in boxes:
            x, y, width, height = box['box']
            x2, y2 = x + width, y + height

            # Preprocess the frame
            input_tensor = preprocess_frame(frame).unsqueeze(0)  # Add batch dimension

            # Predict emotion using PyTorch
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            # Map predicted class to standard emotion
            standard_emotion = class_index_to_emotion.get(predicted_class, 'neutral')

            # Map the standard emotion to a classroom emotion (pass as a list)
            classroom_emotion, intensity = map_standard_to_classroom([standard_emotion])

            # Get the color for the classroom emotion
            color = emotion_colors.get(classroom_emotion, (255, 255, 255))  # Default to white if not in the dict

            # Draw bounding box and display emotion label with the color
            x2, y2 = x + width, y + height
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, classroom_emotion, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Process the frame (e.g., emotion detection, etc.)
        cv2.imshow("Monitoring", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to run the monitoring system in a separate thread
def start_thread():
    thread = threading.Thread(target=start_monitoring)
    thread.start()

# Create a simple Tkinter window with a "Start Monitoring" button
window = tk.Tk()
window.title("Emotion Recognition System")

# Customize the window
window.configure(bg='#4292c6')  # Set background color of window
window.attributes('-fullscreen', True)  # Fullscreen mode

# Create a title label with custom font, size, and color
title_label = tk.Label(
    window, 
    text="Real-time Student Engagement Monitoring System", 
    font=("Helvetica", 24, "bold"),  # Custom font, size, and style
    fg='#61dafb',  # Font color
    bg='#282c34',  # Background color to match the window
    pady=20
)
title_label.pack(pady=50)  # Padding to center the button vertically
# title_label.pack()

# Customize the button appearance
start_button = tk.Button(
    window, 
    text="Start Monitoring", 
    command=start_thread, 
    font=("Helvetica", 16, "bold"),  # Custom button font
    bg='#98c379',  # Button background color
    fg='white',  # Button text color
    activebackground='#61dafb',  # Button background color when clicked
    activeforeground='black',  # Button text color when clicked
    width=20, 
    height=2,
    bd=5  # Border thickness
)
start_button.pack(pady=100)  # Padding to center the button vertically

# Add exit instruction and bind 'Esc' key to exit full-screen
exit_label = tk.Label(
    window, 
    text="Press Esc to exit full-screen", 
    font=("Helvetica", 12), 
    fg='#61dafb', 
    bg='#282c34'
)
exit_label.pack(pady=10)

window.bind("<Escape>", lambda event: window.attributes("-fullscreen", False))


# Run the Tkinter main loop
window.mainloop()
