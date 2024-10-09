import os
import pandas as pd

# Define Standard Emotions and their corresponding Action Units (AUs)
standard_emotions_AU = {
    'happiness': {6, 12},
    'sadness': {1, 4, 15},
    'anger': {4, 5, 7, 23},
    'surprise': {1, 2, 5, 26},
    'fear': {1, 2, 4, 5, 7, 20, 26},
    'disgust': {9, 10, 15, 16, 17},
    'contempt': {12, 14},
    'repression': {4, 12, 14, 15, 17},
    'tense': {4, 5, 6, 7, 12, 14, 17, 24},
    'neutral': {}
}

# Define classroom emotions and their corresponding Action Units (AUs)
classroom_emotions_AU = {
    'focus': {2, 4, 14, 23},
    'distraction': {41, 42, 43, 54, 64},   
    'confusion': {1, 4, 7, 12, 17},
    'frustration': {1, 2, 4, 12, 14},
    'boredom': {4, 7, 12, 17, 43}
}

# Function to calculate the best matching classroom emotion
def map_standard_to_classroom(sequence_of_standard_emotions):
    # Initialize cumulative sum of AUs 
    cumulative_AUs = {}

    # For each standard emotion in the sequence, get the AUs for the standard emotion
    for standard_emotion in sequence_of_standard_emotions:
        AUs = standard_emotions_AU.get(standard_emotion, set())

        # Update the cumulative AUs count
        for action_unit in AUs:
            cumulative_AUs[action_unit] = cumulative_AUs.get(action_unit, 0) + 1

    # Find the classroom emotion with the maximum AU overlap
    matching_classroom_emotion = None
    max_overlap = -1

    for classroom_emotion, classroom_AUs in classroom_emotions_AU.items():
        # Calculate the overlap between cumulative AU and classroom emotion's AUs
        overlap = sum(cumulative_AUs.get(action_unit, 0) for action_unit in classroom_AUs)

        # Update the best matching classroom emotion if current overlap is greater than the previous maximum
        if overlap > max_overlap:
            max_overlap = overlap
            matching_classroom_emotion = classroom_emotion

    return matching_classroom_emotion, max_overlap

emotion_folder_path = "./main/prediction"
emotion_csv_path = os.path.join(emotion_folder_path, 'emotion_predictions_midhalf.csv')

def process_predicted_emotions(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the 'Predicted Emotion' column and convert it to a list
    predicted_emotions = df['Predicted Emotion'].dropna().tolist()

    # Initialize a list to store the group of emotions and their corresponding classroom emotion
    emotion_mapping = []

    # Iterate over predicted emotions in groups of 5
    for i in range(0, len(predicted_emotions), 5):
        # Get the sequence of 5 standard emotions
        sequence_of_standard_emotions = predicted_emotions[i:i + 5]

        # Map the sequence of standard emotions to a classroom emotion
        classroom_emotion, intensity = map_standard_to_classroom(sequence_of_standard_emotions)

        # Store the mapping in the emotion_mapping list
        emotion_mapping.append({
            'sequence_of_standard_emotions': sequence_of_standard_emotions,
            'classroom_emotion': classroom_emotion,
            'intensity': intensity
        })

    return emotion_mapping

# if __name__ == '__main__':
#     # Process the predicted emotions and get the classroom emotion
#     emotion_mapping = process_predicted_emotions(emotion_csv_path)
#     # Print the results
#     for mapping in emotion_mapping:
#         print(f"Standard Emotions: {mapping['sequence_of_standard_emotions']} -> Classroom Emotion: {mapping['classroom_emotion']} (Intensity: {mapping['intensity']})")
