import detect_faces
import os
# import dbcann
import kmeans_and_rest
from make_pickle_files import make_pickle_files_for_respective_folder



all_photos = 'data_new/train_dataset_new'
cropped_faces = 'data_new/cropped_faces_new'
# step one detect faces and store them in a folder
# for photos in os.listdir(all_photos):
#     detect_faces.detect_and_crop_faces(os.path.join(all_photos, photos), cropped_faces)
# # detect_faces.detect_and_crop_faces(all_photos, cropped_faces)
 
# step two create pickle files and cluster ( dbscann part )
input_folder = cropped_faces
output_folder = 'data_new/new_dbscann_jitter_1new_eps0.34_large_final_new'

kmeans_and_rest.cluster_faces(input_folder, output_folder)

######
# generating pickle files
for folder in os.listdir(output_folder):
    for image_file in os.listdir(os.path.join(output_folder, folder)):
        image_path = os.path.join(output_folder, folder, image_file)
        make_pickle_files_for_respective_folder(image_path, os.path.join(output_folder, folder))

# ######
# new_image = 'data_new/collin_powell_test_image.jpeg'
# # step four find the closest match
# import cosine_sim
# cosine_sim.find_closest_match(new_image, 'data_new/Data/train_dataset_pickle_files_new')

 
