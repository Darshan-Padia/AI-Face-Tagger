import os
import shutil
import pickle
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import numpy as np
import face_recognition


# def extract_face_encodings(image_path):
#     """
#     Extract face encodings from an image using face_recognition library.
#     """
#     image = face_recognition.load_image_file(image_path)
#     face_encodings = face_recognition.face_encodings(image)
#     return face_encodings

def get_or_compute_face_embeddings(image_path, embeddings_folder):
    """
    Get or compute face embeddings using DeepFace.
    """
    # Check if embeddings are already computed and stored
    embedding_file = os.path.join(embeddings_folder, os.path.basename(image_path) + ".pkl")
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        # Use DeepFace to get face embeddings
        embedding_objs = DeepFace.represent(img_path=image_path, model_name="Facenet512", enforce_detection=False)
        embeddings = [embedding_obj["embedding"] for embedding_obj in embedding_objs]

        # Store the embeddings for future use
        # if not existing folder then creating it
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)

    return embeddings

def cluster_faces(face_encodings, eps=0.3, min_samples=1):
    """
    Cluster face encodings using DBSCAN algorithm.
    """
    X = np.array(face_encodings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(X)
    return clustering.labels_

def group_photos_by_cluster(photo_paths, cluster_labels):
    """
    Group photos by their cluster labels.
    """
    grouped_photos = {}
    for photo_path, cluster_label in zip(photo_paths, cluster_labels):
        if cluster_label not in grouped_photos:
            grouped_photos[cluster_label] = []
        grouped_photos[cluster_label].append(photo_path)
    return grouped_photos

def move_grouped_photos(grouped_photos, output_folder):
    """
    Move grouped photos to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for cluster_label, photo_paths in grouped_photos.items():
        cluster_folder = os.path.join(output_folder, f"Cluster_{cluster_label}")
        os.makedirs(cluster_folder, exist_ok=True)
        for photo_path in photo_paths:
            shutil.copy(photo_path, cluster_folder)

def main(all_faces_folder, output_folder, embeddings_folder):
    """
    Main function to perform face clustering and grouping.
    """
    # Get all face photo paths
    all_photo_paths = [os.path.join(all_faces_folder, file) for file in os.listdir(all_faces_folder)]

    # Extract or compute face embeddings from all photos
    all_face_embeddings = [embedding for photo_path in all_photo_paths for embedding in get_or_compute_face_embeddings(photo_path, embeddings_folder)]

    # Cluster face embeddings
    cluster_labels = cluster_faces(all_face_embeddings)

    # Group photos by cluster labels
    grouped_photos = group_photos_by_cluster(all_photo_paths, cluster_labels)

    # Move grouped photos to output folder
    move_grouped_photos(grouped_photos, output_folder)

if __name__ == "__main__":
    all_faces_folder = "all_face_photos"
    output_folder = "grouped_photos"
    embeddings_folder = "face_embeddings"
    main(all_faces_folder, output_folder, embeddings_folder)
