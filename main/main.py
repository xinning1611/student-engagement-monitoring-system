import cv2
import torch
import numpy as np
from torchvision import transforms
from emotion_mapping import map_standard_to_classroom  # import your mapping function
from train_model import CustomCNN
from mtcnn.mtcnn import MTCNN

# Load the trained model
model = CustomCNN()  # Instantiate your model class
model.load_state_dict(torch.load('trained_model.path', weights_only=True) ) 
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

# Function to process the input video
def process_video(input_video_path, output_video_path):
    # Define emotion colors
    emotion_colors = {
        'focus': (0, 255, 0),       # Green
        'frustration': (0, 0, 255), # Red
        'confusion': (255, 0, 0),   # Blue
        'boredom': (0, 255, 255),   # Yellow
        'distracted': (0, 165, 255) # Orange
    }

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces and get bounding boxes
        boxes = mtcnn.detect_faces(frame)

        # Draw bounding boxes and get emotions as needed
        for box in boxes:
            x, y, width, height = box['box']
            x2, y2 = x + width, y + height
            # cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)

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

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to get the most dominant emotion
def get_dominant_emotion(emotion_intensities):
    max_intensity = -1
    dominant_emotion = None

    for emotion, intensity in emotion_intensities.items():
        if intensity > max_intensity:
            max_intensity = intensity
            dominant_emotion = emotion

    return dominant_emotion, max_intensity

# if __name__ == "__main__":
#     input_video_focus1 = "./data/test_data/session_1/focus_1.MP4"  # Input video path
#     output_video_focus1 = "./output/test_data/output_focus_1.mp4"  # Output video path
#     process_video(input_video_focus1, output_video_focus1)
#     print("Processing complete. Output saved to", output_video_focus1)