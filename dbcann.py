# how can i imporve the performance of dbsca clusterin here :

import os
import pickle
import face_recognition
from shutil import copyfile
from sklearn.cluster import DBSCAN
import numpy as np

def load_dict(file_path):
    # Read the data from the text file
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Create a dictionary from the data
    original_photo_dict = {}
    for line in data:
        key, value = line.strip().split(': ')
        original_photo_dict[key] = value
    return original_photo_dict


def get_face_encodings(image_path,embeddings_folder = "embeddings_newdbscan_jitter1_final"):
    embedding_file = os.path.join(embeddings_folder, os.path.basename(image_path) + ".pkl")
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            face_encodings = pickle.load(f)
    else :
        
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations,num_jitters=1,model='large')
        # Store the embeddings for future use
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)
        with open(embedding_file, 'wb') as f:
            pickle.dump(face_encodings, f)
    # print(face_encodings)
    return face_encodings

def cluster_faces(image_folder, output_folder):
    # Create the output folder if it doesn't exist
    cluster_flag = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_face_encodings = []
    all_image_paths = []
    i=0

    # Collect face encodings and corresponding image paths
    for image_path in image_paths:
        print(i)
        i=i+1
        face_encodings = get_face_encodings(image_path)
        if face_encodings:
            all_face_encodings.extend(face_encodings)
            all_image_paths.extend([image_path] * len(face_encodings))

    # Convert the list of face encodings to a NumPy array
    X = np.array(all_face_encodings)

    # Use DBSCAN to cluster faces
    dbscan = DBSCAN(eps=0.34, min_samples=1)
    labels = dbscan.fit_predict(X)
    print()
    print(len(np.unique(labels)))
    # original_photo_dict = load_dict("faces_dict.txt")
    # Create folders for each cluster and copy images
    for cluster_id in np.unique(labels):
        cluster_folder = os.path.join(output_folder, f"photos_g{cluster_id}")

        # Create the cluster folder if it doesn't exist
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        # Copy images corresponding to the cluster to the cluster folder
        cluster_indices = np.where(labels == cluster_id)[0] 
        for index in cluster_indices:
            image_path = all_image_paths[index]
            # print(image_path)
            filename = os.path.basename(image_path)
            # if (cluster_flag < 2):
            #     # copying the cropped face with name 'identify'
            #     copyfile(image_path, os.path.join(cluster_folder, f"0identify{cluster_flag}.jpg"))

            #     while not os.path.exists(os.path.join(cluster_folder, f"0identify{cluster_flag}.jpg")):
            #         pass
                
        
            copyfile(image_path, os.path.join(cluster_folder, filename))
            while not os.path.exists(os.path.join(cluster_folder, filename)):
                pass
            cluster_flag+=1

            # copyfile(original_photo_dict[image_path], os.path.join(cluster_folder, filename))
            # wait until file is copied
            # while not os.path.exists(os.path.join(cluster_folder, filename)):
            #     pass
            # while not os.path.exists(os.path.join(cluster_folder, f"0identify{cluster_flag}.jpg")):
            print(f"Image {filename} copied to cluster folder {cluster_folder}")
        cluster_flag = 0

if __name__ == "__main__":
    input_folder = "all_face_photos"  # Change this to your input folder
    output_folder = "new_dbscann_jitter_1new_eps0.34_large_final"  # Change this to your desired output folder

    cluster_faces(input_folder, output_folder)
