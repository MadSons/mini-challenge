import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
import sys
import cv2
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets



temp_folder = 'temp'
pic = '3'
file = f'test_split/{pic}/{pic}.jpg'
path = os.path.join(temp_folder, pic)
print(file)
print(path)

os.makedirs(temp_folder, exist_ok=True)
os.makedirs(path, exist_ok=True)

shutil.copy(file, path)


# Load the data
transforms_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(temp_folder, transforms_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100) # multi-class classification (num_of_class == 100)
model.load_state_dict(torch.load('celebrity_model_1.pth'))

model.eval()
start_time = time.time()

categories = pd.read_csv('category.csv')
categories = categories.rename(columns={'Unnamed: 0': 'id'})
categories_dict = categories.set_index('id')['Category'].to_dict()


with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for i, pred in enumerate(preds):
            prediction = categories_dict[pred.item()]
            print(f'Prediction: {prediction}')

plt.imshow(cv2.imread(file))
plt.show()

shutil.rmtree(temp_folder)
