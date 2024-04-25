import os
from deepface import DeepFace
import cv2
import numpy as np

# Function to detect faces in an image and crop them
# def detect_and_crop_faces(image_path, output_folder):
    
#     try:
#         if(not os.path.exists(output_folder)):
#             os.makedirs(output_folder)
#         # Detect faces in the image using RetinaFace (default)
#         result = DeepFace.extract_faces(image_path, detector_backend='retinaface')
#         # Load the original image
#         img = cv2.imread(image_path)
#         # Save the cropped face
#         for i, face in enumerate(result):
#             x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
#             cropped_face = img[y:y+h, x:x+w]
#             output_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i+1}.jpg")
#             cv2.imwrite(output_filename, cropped_face)
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")

# Function to detect faces in an image and crop them
def detect_and_crop_faces(image_path, output_folder):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Detect faces in the image using RetinaFace (default)
        result = DeepFace.extract_faces(image_path, detector_backend='retinaface')

        # Load the original image
        img = cv2.imread(image_path)

        # Save the cropped face
        for i, face in enumerate(result):
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']

            # Adjust the cropping area to include more surrounding area
            padding = int(0.2 * max(w, h))  # Adjust the padding factor as needed
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img.shape[1], x + w + padding)
            y_max = min(img.shape[0], y + h + padding)

            cropped_face = img[y_min:y_max, x_min:x_max]
            if not os.path.exists(os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0])):
                os.makedirs(os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0]))
            output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0],f"{os.path.splitext(os.path.basename(image_path))[0]}.{i+1}.jpg")
            cv2.imwrite(output_filename, cropped_face)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# main_folder = 'train_dataset'
# input_folders =[]
# for folder in os.listdir(main_folder):
#     input_folders.append(os.path.join(main_folder, folder))

# for input_folder in input_folders:
#     output_folder = os.path.join('cropped_faces', os.path.basename(input_folder))
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     for image_file in os.listdir(input_folder):
#         image_path = os.path.join(input_folder, image_file)
#         detect_and_crop_faces(image_path, output_folder)
# print("Face detection and cropping completed successfully.")