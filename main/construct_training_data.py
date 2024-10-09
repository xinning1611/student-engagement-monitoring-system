# Import statement 
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math
from PIL import Image
import time
import math
# %matplotlib inline

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

# Define data directory (MODIFY THE DIRECTORY TO CHANGE TRAINING SAMPLE!!!)
# data_dir = r".\student-engagement-monitoring-system\data\processed_sample3"
data_dir = "./data/processed_sample3"

# Apply necessary augmentations
transform = transforms.Compose([
    transforms.Grayscale(),                       # Convert images to grayscale
    transforms.RandomHorizontalFlip(),            # Flip the image horizontally
    # transforms.RandomRotation(4),               # Optionally rotate the image
    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Optional affine transformations
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalize for grayscale (mean=0.5, std=0.5 for one channel)
])

# Load the dataset using torchvision.datasets.ImageFolder and apply transformations
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split the dataset into train, validation and test sets
train_size = int(0.8 * len(dataset)) # 80% for train 
valid_size = int(0.1 * len(dataset)) # 10% for validation
test_size = len(dataset) - train_size - valid_size # Remaining 10% for test

# Randomly split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# Create DataLoaders for each of the datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print("Number of instance in train_set: %s" % len(train_dataset))
# print("Number of instance in val_set: %s" % len(val_dataset))
# print("Number of instance in test_set: %s" % len(test_dataset))

# Define emotion class names 
class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 
               'repression', 'sadness', 'surprise', 'tense']

# Function to show the images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# Function to visualize a batch of images 
def visualize_data(images, categories, images_per_row = 8):
    class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 
               'repression', 'sadness', 'surprise', 'tense']
    n_images = len(images)
    n_rows = math.ceil(float(n_images)/images_per_row)
    fig = plt.figure(figsize=(1.5*images_per_row, 1.5*n_rows))
    fig.patch.set_facecolor('white')
    for i in range(n_images):
        plt.subplot(n_rows, images_per_row, i+1)
        plt.xticks([])
        plt.yticks([])
        imshow(images[i])
        class_index = categories[i]
        plt.xlabel(class_names[class_index])
    plt.show()


################ Code to visualize a batch of sample data ####################
# Obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
# visualize_data(images, labels)    # UNCOMMENT TO RUN

# Observe the number of images for each class 
count_class = {}
for _, outs in dataset:
    labels = class_names[outs]
    if labels not in count_class:
        count_class[labels] = 0
    count_class[labels] += 1

# print(count_class) # UNCOMMENT TO RUN
# {'anger': 55, 'contempt': 12, 'disgust': 68, 'fear': 10, 'happiness': 59, 
# 'repression': 27, 'sadness': 13, 'surprise': 39, 'tense': 103}
