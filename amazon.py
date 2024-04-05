import boto3
import os
from PIL import Image
import sys
import numpy as np
import cv2
import shutil
import time

# Set up AWS Rekognition client
rekognition = boto3.client('rekognition')

# Set the input and output folders
input_folder = 'train'
output_folder = 'train_faces_amazon'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

no_face = 0
start = time.time()
extreme_error_files = []
# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # Open the image
        image_path = os.path.join(input_folder, filename)
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Use Rekognition to detect faces
        try:
            response = rekognition.detect_faces(Image={'Bytes': image_bytes})
        except:
            print('Error in', image_path)
            try:
                img = Image.open(image_path).convert('RGB')
                os.remove(image_path)
                img.save(image_path)
                with open(image_path, 'rb') as image_file:
                    image_bytes = image_file.read()
                response = rekognition.detect_faces(Image={'Bytes': image_bytes})
            except:
                print('LARGE ERROR in', image_path)
                extreme_error_files.append(image_path)
                img = Image.open(image_path).convert('RGB')
                os.remove(image_path)
                img = img.resize((1028, 1028))
                img.save(image_path)
                with open(image_path, 'rb') as image_file:
                    image_bytes = image_file.read()
                response = rekognition.detect_faces(Image={'Bytes': image_bytes})

        # If faces are detected, draw bounding boxes and save the image
        if response['FaceDetails']:
            image = Image.open(image_path).convert('RGB')
            faces = []

            for face in response['FaceDetails']:
                
                box = face['BoundingBox']
                left = image.width * box['Left']
                top = image.height * box['Top']
                right = image.width * box['Left'] + image.width * box['Width']
                bottom = image.height * box['Top'] + image.height * box['Height']
                faces.append([left, top, right, bottom])

            largest_face = max(faces, key=lambda f: (f[0] - f[2]) * (f[1] - f[3]))
            left, top, right, bottom = largest_face
            x, y = int((left + right) / 2), int((top + bottom) / 2)
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            # create a square centered at x, y with side length as the larger of right - left and bottom - top
            side = max(right - left, bottom - top)
            left = x - side // 2
            top = y - side // 2
            right = x + side // 2
            bottom = y + side // 2

            # if negative, set to 0
            left = max(0, left)
            top = max(0, top)

            img = np.array(image)

            try:
                img = cv2.cvtColor(img[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            except:
                print('Error in', filename)
                print(img)
                print(image.width, image.height)
                print(img.shape)
                print(left, top, right, bottom)
                extreme_error_files.append(image_path + 'color error')
                continue

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)

        else:
            print('No faces detected in', image_path)
            no_face += 1

            # write the image to the no_faces folder
            no_face_dir = 'no_faces_train/'
            os.makedirs(no_face_dir, exist_ok=True)
            shutil.copyfile(image_path, no_face_dir + filename)
            
print('Total images:', len(os.listdir(input_folder)))
print('Images with no faces:', no_face)
print('Images with faces:', len(os.listdir(output_folder)))
print('Percentage of images with faces:', len(os.listdir(output_folder)) / len(os.listdir(input_folder)) * 100, '%')
print('Time taken:', time.time() - start, 'seconds')
print('EXTREME ERROR FILES:', extreme_error_files)