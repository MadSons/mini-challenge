import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
import sys
import cv2
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.optim.lr_scheduler import ExponentialLR

# Enable GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the data
transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]) # normalization
])

transforms_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
])

train_dataset = datasets.ImageFolder('train_amazon_split', transforms_train)
test_dataset = datasets.ImageFolder('val_amazon_split', transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

class_names = train_dataset.classes
print('Class names:', class_names)

# Define the model
model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100) # multi-class classification (num_of_class == 100)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.1)

# Train the model
num_epochs = 50

train_losses = []
train_accs = []
val_losses = []
val_accs = []

start_time = time.time()

for epoch in range(1, num_epochs+1):
    """ Training Phase """
    model.train()

    running_loss = 0.
    running_corrects = 0

    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # get loss value and update the network weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Update the learning rate
    if epoch % 10 == 0:
        scheduler.step()

    # Calculate the loss and accuracy
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.item())
    print('[Epoch {}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time), end = ' ')

    # Save the model
    name = f'celebrity_model_10_epoch_{epoch}.pth'
    torch.save(model.state_dict(), name)
    """ Test Phase """
    model.eval()

    # Validation
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        print('Val Loss: {:.4f} Val Acc: {:.4f}% Time: {:.4f}s lr: {:.4f}'.format(epoch_loss, epoch_acc, time.time() - start_time, optimizer.param_groups[0]['lr']))


# Save the model
name = 'celebrity_model_10.pth'
torch.save(model.state_dict(), name)

# Plot the loss and accuracy
plt.figure(figsize=(12, 8))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title(f'Loss for {name} with Epochs {num_epochs}, opt=SGD, lr=0.01')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title(f'Accuracy for {name} with Epochs {num_epochs}, opt=SGD, lr=0.01')
plt.legend()
plt.show()

