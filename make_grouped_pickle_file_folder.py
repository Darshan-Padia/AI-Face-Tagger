import os
from shutil import copyfile

def make_grouped_pickle_file_folder(pickle_folder_path, grouped_photos_folder_path):
    
    grouped_folders_inside = os.listdir(grouped_photos_folder_path)
    print(grouped_folders_inside)
    # create folder goruped_pickle_files if not exists
    if not os.path.exists("grouped_pickle_files"):
        os.makedirs("grouped_pickle_files")
    for grouped_folder in grouped_folders_inside:
        for image_file in os.listdir(os.path.join(grouped_photos_folder_path, grouped_folder)):
            pickle_file_path = os.path.join(pickle_folder_path, image_file + ".pkl")
            destination_folder = os.path.join("grouped_pickle_files", grouped_folder)
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            copyfile(pickle_file_path, os.path.join(destination_folder, image_file + ".pkl"))
    print("Grouped pickle files created successfully.")
    
    
# Example usage
make_grouped_pickle_file_folder("embeddings_newdbscan_jitter1_final", "new_dbscann_jitter_1new_eps0.34_large_final")
            
            

   