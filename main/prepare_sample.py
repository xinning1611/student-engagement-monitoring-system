import os
import shutil
import pandas as pd
import os

# Define CASMEII and SAMM folder path
casmeii_folder_path = "./data/CASMEII"  # Use forward slashes or raw strings
samm_folder_path = "./data/SAMM"

# Load Excel file of CASMEII and SAMM
casmeii_csv_path = os.path.join(casmeii_folder_path, 'CASME2.csv')
casmeii_csv_data = pd.read_csv(casmeii_csv_path)  # Use pd.read_excel for Excel files
samm_csv_path = os.path.join(samm_folder_path, 'SAMM.csv')
samm_csv_data = pd.read_csv(samm_csv_path, dtype={'Subject': str})  # Use pd.read_excel for Excel files

# Define the base directory where emotion class folders will be saved
save_dir = "./data"
sample2_dir = os.path.join(save_dir, 'sample2')
sample3_dir = os.path.join(save_dir, 'sample3')

# Function to get the mid 3 images from a folder
def get_mid_3_images(img_folder):
    """
    This function returns the middle 3 images from the img_folder
    """
    images = sorted(os.listdir(img_folder))
    if len(images) < 3:
        return images
    mid_idx = len(images) // 2
    return images[mid_idx - 1:mid_idx + 2]

def get_mid_half_images(img_folder):
    """
    Returns the middle half of images from the img_folder
    """
    images = sorted(os.listdir(img_folder))
    
    if len(images) < 2:
        return images  # Return all images if less than 2
    
    # get total number of images and the middle index
    total_images = len(images)
    mid_idx = total_images // 2
    
    # Calculate start and end indices for the middle half
    start_idx = mid_idx - (total_images // 4)
    end_idx = mid_idx + (total_images // 4)
    
    # Ensure indices are within bounds
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, total_images)
    
    return images[start_idx:end_idx]

# Dictionary to keep track of the highest index used for each emotion class
emotion_class_index = {}
# Function to move or copy the selected images
def move_images_to_class_folder(images, src_folder, dest_folder, emotion_class):
    """
    This function organize the images to the specified destination folder
    """
    os.makedirs(dest_folder, exist_ok=True)  # Create the destination folder if it doesn't exist

    # Initialize the index for the emotion class if it doesn't exist
    if emotion_class not in emotion_class_index:
        emotion_class_index[emotion_class] = 1
    
    for image in images:
        # Create the new filename with the global index for the emotion class
        new_image_name = f"{emotion_class.lower()}{str(emotion_class_index[emotion_class]).zfill(2)}.png"
        src_image_path = os.path.join(src_folder, image)
        dest_image_path = os.path.join(dest_folder, new_image_name)
        
        # Copy the image to the destination folder with the new name
        shutil.copy(src_image_path, dest_image_path)
        
        # Increment the index for the emotion class
        emotion_class_index[emotion_class] += 1

# Iterate over each row in the CSV
def prepare_sample2(data, data_folder_path):
    save_dir = "./data"
    sample2_dir = os.path.join(save_dir, 'sample2')

    for index, row in data.iterrows():
        # Construct the folder path using subject and filename (episode)
        subject_str = str(row['Subject'])
        filename_str = str(row['Filename'])
        sub_folder = os.path.join(data_folder_path, subject_str)
        ep_folder = os.path.join(sub_folder, filename_str)
        
        if os.path.exists(ep_folder):
            # Get mid 3 images from the episode folder
            mid_images = get_mid_3_images(ep_folder)
            
            # Define the destination folder inside sample2 based on the emotion class
            dest_folder = os.path.join(sample2_dir, row['Emotion'].lower())
            
            # Move or copy the images to the emotion class folder in sample2 with renaming
            move_images_to_class_folder(mid_images, ep_folder, dest_folder, row['Emotion'].lower())
            
            print(f"Processed {row['Subject']}/{row['Filename']} -> {row['Emotion'].lower()}")
        else:
            print(f"Episode folder {ep_folder} not found")
 
# Dictionary to keep track of the highest index used for each emotion class
emotion_class_index = {}
def prepare_sample3(data, data_folder_path):
    # Iterate over each row in the CSV
    for index, row in data.iterrows():
        # Construct the folder path using subject and filename (episode)
        subject_str = str(row['Subject'])
        filename_str = str(row['Filename'])
        sub_folder = os.path.join(data_folder_path, subject_str)
        ep_folder = os.path.join(sub_folder, filename_str)
        
        if os.path.exists(ep_folder):
            # Get mid 3 images from the episode folder
            mid_images = get_mid_half_images(ep_folder)
            
            # Define the destination folder inside sample3 based on the emotion class
            dest_folder = os.path.join(sample3_dir, row['Emotion'].lower())
            
            # Move or copy the images to the emotion class folder in sample3 with renaming
            move_images_to_class_folder(mid_images, ep_folder, dest_folder, row['Emotion'].lower())
            
            print(f"Processed {row['Subject']}/{row['Filename']} -> {row['Emotion'].lower()}")
        else:
            print(f"Episode folder {ep_folder} not found")


#################### Code to construct samples ####################
# UNCOMMENT TO RUN
# prepare_sample2(casmeii_csv_data, casmeii_folder_path) #747 images
# prepare_sample2(samm_csv_data, samm_folder_path)

# prepare_sample3(casmeii_csv_data, casmeii_folder_path) #8216 images
# prepare_sample3(samm_csv_data, samm_folder_path) 
