# This file mainly handles the preprocessing of the images before feeding into our model
import os
import cv2
import json
import csv

def preprocess_sample_images(dataset, dataset_path, processed_dataset_path):
    image_label_list = []

    # Create a label-to-index mapping
    label_to_index = {label: index for index, label in enumerate(sorted(dataset))}
    index_to_label = {index: label for label, index in label_to_index.items()}

    # Save the label mappings for future reference
    with open(os.path.join(processed_dataset_path, 'label_mappings.json'), 'w') as f:
        json.dump({'label_to_index': label_to_index, 'index_to_label': index_to_label}, f, indent=4)

    for emotion_class in dataset:
        # Construct paths
        emotion_class_path = os.path.join(dataset_path, emotion_class)
        images_path = os.listdir(emotion_class_path)
        processed_image_path = os.path.join(processed_dataset_path, emotion_class)
        os.makedirs(processed_image_path, exist_ok=True)  # Create the destination folder if it doesn't exist
    
        for image_label in images_path:
            image_path = os.path.join(emotion_class_path, image_label)
            # Read the image
            image = cv2.imread(image_path)
            # if the image has 3 channels, means it's a coloured image so convert it to grey scale image
            if len(image.shape) == 3:
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the image to a fixed size of 64 x 64 pixels
            resized_image = cv2.resize(grayscale_image, (64, 64))
            
            # Normalize the image data
            normalized_image = resized_image / 255.0
            
            # Save the processed image
            save_image_path = os.path.join(processed_image_path, image_label)
            cv2.imwrite(save_image_path, (normalized_image * 255).astype('uint8'))

            # Encode the label as an integer
            encoded_label = label_to_index[emotion_class]
            
            # Append the path and label to the list
            image_label_list.append({'image_path': save_image_path, 'label': encoded_label})
            
    # Save the image paths and labels to a CSV file
    with open(os.path.join(processed_dataset_path, 'processed_image.csv'), 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in image_label_list:
            writer.writerow(entry)

    # print(len(image_label_list)) # 386 images in total for sample 1

def process_one_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # if the image has 3 channels, means it's a coloured image so convert it to grey scale image
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size of 64 x 64 pixels
    resized_image = cv2.resize(grayscale_image, (64, 64))
    
    # Normalize the image data
    normalized_image = resized_image / 255.0

    return normalized_image


################ Declare path for storing processed samples ####################
dataset_1_path = "./data/sample"
processed_dataset_1_path = "./data/processed_sample"
dataset_1 = os.listdir(dataset_1_path)
# preprocess_sample_images(dataset_1, dataset_1_path, processed_dataset_1_path)     # UNCOMMENT TO RUN

dataset_2_path = "./data/sample2"
processed_dataset_2_path = "./data/processed_sample2"
dataset_2 = os.listdir(dataset_2_path)
# preprocess_sample_images(dataset_2, dataset_2_path, processed_dataset_2_path)     # UNCOMMENT TO RUN

dataset_3_path = "./data/sample3"
processed_dataset_3_path = "./data/processed_sample3"
dataset_3 = os.listdir(dataset_3_path)
# preprocess_sample_images(dataset_3, dataset_3_path, processed_dataset_3_path)     # UNCOMMENT TO RUN