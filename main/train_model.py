# Import statements 
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
# from construct_training_data import train_loader, val_loader, test_loader, device, class_names  # Import data and device from your data preparation script
# import main.construct_training_data as construct_training_data
import pandas as pd
# from main.construct_training_data import device
# from main.construct_training_data import train_loader, val_loader, test_loader, device, class_names

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

# Define the architecture and operations of our convolutional neural network
class CustomCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        This function performs forward propagation to pass the input through multiple layers and extract features
        """
        # Apply convolutional layers
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 8 * 8 * 128)
        
        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Construct the BaseTrainer class 
class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.stop_training = False

    def fit(self, num_epochs):
        self.num_batches = len(self.train_loader)
        for epoch in range(num_epochs):
            if self.stop_training:
                break
            print(f'Epoch {epoch + 1}/{num_epochs}')
            start_time = time.time()
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, top1_acc, top5_acc = self.validate_one_epoch()
            print(f' train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - top1_acc: {top1_acc:.4f} - top5_acc: {top5_acc:.4f}')
            self.on_epoch_end({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': top1_acc,
                'top5_acc': top5_acc
            })

    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct / total
        train_loss = running_loss / self.num_batches
        return train_loss, train_accuracy

    def validate_one_epoch(self):
        self.model.eval()
        val_loss, correct, total_top1_acc, total_top5_acc, total = 0.0, 0, 0.0, 0.0, 0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                top1_acc = top_k_accuracy(outputs, labels, k=1)
                top5_acc = top_k_accuracy(outputs, labels, k=5)
                total_top1_acc += top1_acc * inputs.size(0)
                total_top5_acc += top5_acc * inputs.size(0)
                total += labels.size(0)
        val_loss /= len(self.val_loader)
        top1_acc = total_top1_acc / total
        top5_acc = total_top5_acc / total
        return val_loss, top1_acc, top5_acc

    def on_epoch_end(self, params):
        pass

# Function to compute loss 
def compute_loss(model, loss_fn, loader):
    loss = 0
    model.eval()    # Set model to eval mode for inference
    
    with torch.no_grad():  # No need to track gradients for validation
        for (batchX, batchY) in loader:
            # Move data to the same device as the model
            batchX, batchY = batchX.to(device).type(torch.float32), batchY.to(device).type(torch.long)
            loss += loss_fn(model(batchX), batchY)
    # Set model back to train mode
    model.train()
    return float(loss)/len(loader)

# Function to compute accuracy
def compute_acc(model, loader):
    correct = 0
    totals = 0
    model.eval()        # Set model to eval mode for inference

    for (batchX, batchY) in loader:
        # Move batchX and batchY to the same device as the model
        batchX, batchY = batchX.to(device).type(torch.float32), batchY.to(device)
        outputs = model(batchX)  # feed batch to the model
        totals += batchY.size(0)  # accumulate totals with the current batch size
        predicted = torch.argmax(outputs.data, 1)  # get the predicted class
        # Move batchY to the same device as predicted for comparison
        correct += (predicted == batchY).sum().item()
    return correct / totals

# Function to train model while computing loss and accuracy 
def fit(model= None, train_loader = None, valid_loader= None, optimizer = None,
        num_epochs = 50, verbose = True):
    """
    This function trains the model while computing its loss and accuracy
    """
    # Move the model to the device before initializing the optimizer
    model.to(device)
    if optimizer == None:
        optim = torch.optim.Adam(model.parameters(), lr = 0.001) # Now initialize optimizer with model on GPU
    else:
        optim = optimizer
    history = dict()
    history['val_loss'] = list()
    history['val_acc'] = list()
    history['train_loss'] = list()
    history['train_acc'] = list()

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        for (X, y) in train_loader:
            # Move input data to the same device as the model
            X,y = X.to(device), y.to(device)
            # Forward pass
            outputs = model(X.type(torch.float32))
            loss = F.cross_entropy(outputs, y.type(torch.long))
            # Backward and optimize
            optim.zero_grad()
            loss.backward()
            optim.step()
        #losses and accuracies for epoch
        val_loss = compute_loss(model, F.cross_entropy, valid_loader)
        val_acc = compute_acc(model, valid_loader)
        train_loss = compute_loss(model, F.cross_entropy, train_loader)
        train_acc = compute_acc(model, train_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        end = time.time()
        #   print(f"total time for each epoch {end - start}") # time in seconds
        if not verbose: #verbose = True means we do show the training information during training
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"train loss= {train_loss:.4f} - train acc= {train_acc*100:.2f}% - valid loss= {val_loss:.4f} - valid acc= {val_acc*100:.2f}%")
    return history


# Function to define the top_k_accuracy for evaluation
def top_k_accuracy(output, target, k=5):
    batch_size = target.size(0)
    _, pred = output.topk(k, 1, True, True)  # Get top-k predictions
    pred = pred.t()  # Transpose predictions for comparison
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))  # Compare predictions with target
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Calculate correct top-k
    return correct_k.mul_(1.0 / batch_size).item()  # Calculate top-k accuracy

# Function to compute accuracy of test data 
def test_accuracy(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct, total_top1_acc, total_top5_acc, total = 0, 0.0, 0.0, 0

    with torch.no_grad():  # Disable gradient computation during inference
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Calculate top-1 and top-5 accuracy
            top1_acc = top_k_accuracy(outputs, labels, k=1)
            top5_acc = top_k_accuracy(outputs, labels, k=5)

            # Accumulate metrics
            total_top1_acc += top1_acc * inputs.size(0)
            total_top5_acc += top5_acc * inputs.size(0)
            total += labels.size(0)

    top1_acc = total_top1_acc / total
    top5_acc = total_top5_acc / total
    print(f"Test Accuracy: Top-1: {top1_acc * 100:.2f}%, Top-5: {top5_acc * 100:.2f}%")
    return top1_acc, top5_acc

# Function to test the model and compare actual labels vs predicted labels
def test_model(model, test_loader, class_names):
    model.eval()  # Set the model to evaluation mode
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get the model's predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Append actual and predicted labels
            actual_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Map numeric labels to class names
    actual_emotions = [class_names[label] for label in actual_labels]
    predicted_emotions = [class_names[label] for label in predicted_labels]
    
    # Create a DataFrame with actual and predicted labels
    results_df = pd.DataFrame({
        'Actual Emotion': actual_emotions,
        'Predicted Emotion': predicted_emotions
    })

    # Save the DataFrame to a CSV file
    # results_df.to_csv('student-engagement-monitoring-system/main/prediction/emotion_predictions_midhalf.csv', 
                    #   index=False)
    results_df.to_csv('prediction/emotion_predictions_midhalf.csv', index=False)
    print("Test results saved to 'emotion_predictions_midhalf.csv'.")


###################### CODE FOR TRAINING MODEL #############################
# Define the dictionary of optimizers 
optim_dict = {"Adam":optim.Adam, "Adadelta":optim.Adadelta, "Adagrad":optim.Adagrad,
              "Adamax":optim.Adamax, "AdamW": optim.AdamW, "ASGD":optim.ASGD,
              "NAdam":optim.NAdam, "RMSprop":optim.RMSprop, "RAdam":optim.RAdam,
              "Rprop": optim.Rprop, "SGD":optim.SGD}
   
# Initialize a model instance
model = CustomCNN(num_classes=9).to(device)

# Declare loss and optimizer
learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = optim_dict["Adam"](model.parameters(), lr=learning_rate)

# Train the model, compute train and validation loss and accuracy (COMMENT OUT AFTER TRAIN MODEL!!!)
# history = fit(model = model, train_loader = train_loader, valid_loader = val_loader,
    # optimizer = optimizer, num_epochs= 20, verbose = False)

# save the trained model (COMMENT OUT AFTER TRAIN MODEL!!!)
# torch.save(model.state_dict(), 'trained_model.path')

# # Test against test data
# top1_acc, top5_acc = test_accuracy(model, test_loader, device)
# Call the function to test the model and save the results
# test_model(model, test_loader, class_names)

