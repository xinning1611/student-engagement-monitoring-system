import cv2
import os

def video_to_frames(video_path, output_folder):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get the frame rate of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")
    
    frame_count = 0
    while True:
        # Read a frame from the video
        success, frame = video_capture.read()
        
        if not success:
            break  # Break when the video ends
        
        # Save the current frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()
    print(f"Total frames extracted: {frame_count}")

# Example usage:
video_path = "./output/output_video1_colour_code.mp4"
output_folder = "./output/output_frames"
video_to_frames(video_path, output_folder)
