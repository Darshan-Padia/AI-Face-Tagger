# working
from deepface import DeepFace
import os
import shutil

# Directory paths for your photos
target_photos_dir = "target"
all_photos_dir = "all_photos"

matching_photos_dir = "matching_photos"  # Folder for matched photos
target_photos_files = os.listdir(target_photos_dir)
i=0
# Loop through each file in the target_photos folder and compare with all photos
for target_photo_file in target_photos_files:
    target_photo_path = os.path.join(target_photos_dir, target_photo_file)

    all_photos_files = os.listdir(all_photos_dir)

    for all_photo_file in all_photos_files:
        all_photo_path = os.path.join(all_photos_dir, all_photo_file)

        try:
            result = DeepFace.verify(
                img1_path=target_photo_path,
                img2_path=all_photo_path,
                model_name="Facenet512",
                detector_backend="retinaface",
                distance_metric="euclidean_l2",
                enforce_detection=False  # Set enforce_detection to False
            )

            # Check if the verification result indicates a match
            # print(result["distance"])
            if result["distance"] < 0.915:
            #   print()
            #   i+=1
            #   print(i)
            #   print(result)
              print(f" {target_photo_file} and {all_photo_file}")

                # Copy the matched photos to the matching_photos folder
            #   shutil.copy(all_photo_path, os.path.join(matching_photos_dir, all_photo_file))

        except ValueError as e:
            # Handle the case where a face was not detected in one of the images
            # print(f"Error: {e}")
            pass