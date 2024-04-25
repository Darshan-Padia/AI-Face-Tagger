import os
import pickle
import numpy as np
import face_recognition
# Function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def make_pickle_files(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations,num_jitters=1,model='large')
    # print(face_encodings[0])
    return face_encodings  



# problem with cosine similarity logic is that, lets say by chance while clustering person1's any image is in other cluster other than its own,thenit might return wrong cluster
# to fix this we have 2 approaches 
# 1 is AI model train ( images -> cnn )
# 2 is find max number of matches after setting certain threshold and returning the cluster with max number of matches..

# Approach 2
def find_closest_match_majority(new_image_path, grouped_faces_dir_with_pickle_files,threashold=0.9):
    new_image_feature = make_pickle_files(new_image_path)[0]
    max_similarity = -1.0
    folder_with_max_matches = None
    max_similarity_folder = None
    max_matches = 0
    # threashold = 0.8
    # we will implement this logic with two factor authentication kind of thing, we will
    # keep track of max similarity as well as second max similartiy(not as of now ) and max_matches
    # if majority of the matches in an cluster are above threashold and its the same folder as max_similarity_folder
    # then we will return that folder as the closest match
    # but if the majority of the matches are above threashold but in different folder than the max_similarity_folder 
    # then we will return both the folders as the closest match
    for folder_name in os.listdir(grouped_faces_dir_with_pickle_files):
        folder_path = os.path.join(grouped_faces_dir_with_pickle_files, folder_name)
        if os.path.isdir(folder_path):
            matches = 0
            for file in os.listdir(folder_path):
                if(file.endswith('.pkl')):
                    pickle_file = file
                else:
                    continue
                pickle_file_path = os.path.join(folder_path, pickle_file)
                with open(pickle_file_path, 'rb') as f:
                    existing_features = pickle.load(f)
                    
                    # print("existing_features",existing_features[0])
                    if(existing_features != []):
                        feature = existing_features[0]
                        similarity = cosine_similarity(new_image_feature, feature)
                        if similarity > threashold:
                            matches += 1
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_similarity_folder = folder_name
                            
            if matches > max_matches:
                max_matches = matches
                folder_with_max_matches = folder_name
    if folder_with_max_matches == max_similarity_folder:
        print(f"The closest match is in the folder: {folder_with_max_matches}")
        # returning folder path 
        return os.path.join(grouped_faces_dir_with_pickle_files, folder_with_max_matches)
    elif(max_matches > 0):
        print(f"The closest matches can be the folders: {folder_with_max_matches} and {max_similarity_folder}")
        return os.path.join(grouped_faces_dir_with_pickle_files, folder_with_max_matches)
    else:
        print("No match found.")
    


def find_closest_match(new_image_path, grouped_faces_dir_with_pickle_files):
    new_image_feature = make_pickle_files(new_image_path)[0]
    max_similarity = -1.0
    closest_match_folder = None
    for folder_name in os.listdir(grouped_faces_dir_with_pickle_files):
        folder_path = os.path.join(grouped_faces_dir_with_pickle_files, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if(file.endswith('.pkl')):
                    pickle_file = file
                else:
                    continue
                pickle_file_path = os.path.join(folder_path, pickle_file)
                with open(pickle_file_path, 'rb') as f:
                    existing_features = pickle.load(f)
                    if(existing_features != []):
                        similarities = [cosine_similarity(new_image_feature, feature) for feature in existing_features]
                        current_max_similarity = max(similarities)
                        if current_max_similarity > max_similarity:
                            max_similarity = current_max_similarity
                            closest_match_folder = folder_name
    if closest_match_folder:
        print(f"The closest match is in the folder: {closest_match_folder}")
    else:
        print("No match found.")


# new_image_path = 'collin_powell_test_image.jpeg'
# new_image_feature =  make_pickle_files(new_image_path)[0]

# # Path to the 'grouped_pickle_files' directory
# pickle_files_dir = 'Data/train_dataset_pickle_files'

# max_similarity = -1.0
# closest_match_folder = None

# # Iterate over folders in the 'grouped_pickle_files' directory
# for folder_name in os.listdir(pickle_files_dir):
#     folder_path = os.path.join(pickle_files_dir, folder_name)
#     # print(folder_path)
#     if os.path.isdir(folder_path):
#         # Iterate over pickle files in the current folder
#         for pickle_file in os.listdir(folder_path):
            
#             # print('Data/train_dataset_pickle_files/George_W_Bush/George_W_Bush_0001_1.pkl')
#             # print("from the code : ",os.path.join(folder_path, pickle_file))
#             pickle_file_path = os.path.join(folder_path, pickle_file)
#             with open(pickle_file_path, 'rb') as f:
#                 existing_features = pickle.load(f)
#                 if(existing_features != []):
#                 # print('existing_features')
#                 # print(existing_features)
#                 # Calculate cosine similarity between the new image and each feature vector in the pickle file
#                     similarities = [cosine_similarity(new_image_feature, feature) for feature in existing_features]
#                     current_max_similarity = max(similarities)
#                     if current_max_similarity > max_similarity:
#                         max_similarity = current_max_similarity
#                         closest_match_folder = folder_name

# if closest_match_folder:
#     print(f"The closest match is in the folder: {closest_match_folder}")
# else:
#     print("No match found.")