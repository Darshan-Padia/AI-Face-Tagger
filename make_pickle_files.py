import os
import pickle
import face_recognition
import shutil
from sklearn.cluster import DBSCAN
import numpy as np


def make_pickle_files(image_path,sub_folder,embeddings_folder):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations,num_jitters=1,model='large')
    # Store the embeddings for future use
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    # make subfolder for each person
    if not os.path.exists(os.path.join(embeddings_folder, os.path.basename(sub_folder))):
        os.makedirs(os.path.join(embeddings_folder, os.path.basename(sub_folder)))
    embedding_file = os.path.join(embeddings_folder, os.path.basename(sub_folder), os.path.splitext(os.path.basename(image_path))[0] + ".pkl")
    with open(embedding_file, 'wb') as f:
        pickle.dump(face_encodings, f)

# making and saving pickle files for each image in its respective folder

def make_pickle_files_for_respective_folder(image_path,image_folder):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations,num_jitters=1,model='large')
    # in which folder where the image is present
    embeddings_folder = image_folder
    # Store the embeddings for future use
    embedding_file = os.path.join(embeddings_folder, os.path.splitext(os.path.basename(image_path))[0] + ".pkl")
    with open(embedding_file, 'wb') as f:
        pickle.dump(face_encodings, f)
        cropped_faces_folder =  os.path.splitext(os.path.basename(image_path))[0].split('.')[0]
        
    cropped_faces_path = os.path.join("data_new/cropped_faces_new",cropped_faces_folder)
    # copy the pickle file in the cropped_faces_path folder
    shutil.copy(embedding_file, cropped_faces_path)
    
    
    
        
    
# def make_pickle_files_for_respective_folder_cropped_faces(image_path,image_folder):
#     image = face_recognition.load_image_file(image_path)
#     face_locations = face_recognition.face_locations(image)
#     face_encodings = face_recognition.face_encodings(image, face_locations,num_jitters=1,model='large')
#     # in which folder where the image is present
#     embeddings_folder = image_folder
#     # Store the embeddings for future use
#     embedding_file = os.path.join(embeddings_folder, os.path.splitext(os.path.basename(image_path))[0] + ".pkl")
#     with open(embedding_file, 'wb') as f:
#         pickle.dump(face_encodings, f)
        

        
# main_folder = 'resized_faces'

# sub_folders =[]

# for folder in os.listdir(main_folder):
#     sub_folders.append(os.path.join(main_folder, folder))
    
# for sub_folder in sub_folders:
#     embeddings_folder = os.path.join('train_dataset_pickle_files', os.path.basename(sub_folder))
#     if not os.path.exists(embeddings_folder):
#         os.makedirs(embeddings_folder)
#     for image_file in os.listdir(sub_folder):
#         image_path = os.path.join(sub_folder, image_file)
#         make_pickle_files(image_path,sub_folder,embeddings_folder)
        
        