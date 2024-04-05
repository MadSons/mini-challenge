import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
import shutil

from sklearn.model_selection import train_test_split
import cv2

# Read the CSV file
data = pd.read_csv('train.csv')
src = 'train_faces_amazon'

# Remove rows with missing files from the data dataframe
data = data[data['File Name'].isin(os.listdir(src))]

# get id for class names
categories = pd.read_csv('category.csv')
categories = categories.rename(columns={'Unnamed: 0': 'id'})
categories_dict = categories.set_index('Category')['id'].to_dict()

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.025, random_state=42)

# Create the train and val folders
train_folder = 'train_amazon_split'
val_folder = 'val_amazon_split'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Move the training images to the train folder
for _, row in train_data.iterrows():
    class_name = str(row['Category'])
    file_name = str(row['File Name'])
    source_path = os.path.join(src, file_name)
    destination_path = os.path.join(train_folder, str(categories_dict[class_name]), file_name)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copyfile(source_path, destination_path)

# Move the validation images to the val folder
for _, row in val_data.iterrows():
    class_name = str(row['Category'])
    file_name = str(row['File Name'])
    source_path = os.path.join(src, file_name)
    destination_path = os.path.join(val_folder, str(categories_dict[class_name]), file_name)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copyfile(source_path, destination_path)