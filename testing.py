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

# Enable GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([1, 1, 1])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

# Load the data
transforms_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
])

test_dataset = datasets.ImageFolder('test_amazon_split', transforms_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)

print('Test dataset size:', len(test_dataset))
id_names = test_dataset.classes
#print('Class names:', class_names)


model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100) # multi-class classification (num_of_class == 100)
model.load_state_dict(torch.load('celebrity_model_10_epoch_35.pth'))
model.to(device)

model.eval()
start_time = time.time()

categories = pd.read_csv('category.csv')
categories = categories.rename(columns={'Unnamed: 0': 'id'})
categories_dict = categories.set_index('id')['Category'].to_dict() # integer to string

f = open('celebrity_result_10.csv', 'w')
f.write('Id,Category\n')
predictions = {}
class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        """print(preds)
        print(labels)
        print([class_names[x] for x in preds])"""
        for j, x in enumerate(preds):
            predictions[int(id_names[labels[j].item()])] = categories_dict[int(class_names[x])]

        if i == 0:
            images = torchvision.utils.make_grid(inputs[:8])
            imshow(images.cpu(), title=[categories_dict[int(class_names[x])] for x in preds[:8]])
        """print(predictions)
        images = torchvision.utils.make_grid(inputs)
        imshow(images.cpu(), title=[categories_dict[int(class_names[x])] for x in preds])"""       
        
for key in sorted(predictions.keys()):
   f.write(f'{key},{predictions[key]}\n')

f.close()

print('Time taken:', time.time() - start_time)