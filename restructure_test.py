import matplotlib.pyplot as plt
import os
import shutil


files = os.listdir('test_faces_amazon')
print(len(files))
for filename in files:
    source_path = os.path.join('test_faces_amazon', filename)
    index = filename.split('.')[0]
    destination_path = os.path.join('test_amazon_split', index, filename)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copyfile(source_path, destination_path)