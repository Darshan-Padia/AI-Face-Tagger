# def get_or_compute_face_embeddings(image_path, embeddings_folder):
#     # Check if embeddings are already computed and stored
#     embedding_file = os.path.join(embeddings_folder, os.path.basename(image_path) + ".pkl")
#     if os.path.exists(embedding_file):
#         with open(embedding_file, 'rb') as f:
#             embeddings = pickle.load(f)
#     else:
#         # Use DeepFace to get face embeddings
#         embedding_objs = DeepFace.represent(img_path=image_path, model_name="Facenet512", enforce_detection=False)
#         embeddings = [embedding_obj["embedding"] for embedding_obj in embedding_objs]

#         # Store the embeddings for future use
#         # if not existing folder then creating it
#         if not os.path.exists(embeddings_folder):
#             os.makedirs(embeddings_folder)
#         with open(embedding_file, 'wb') as f:
#             pickle.dump(embeddings, f)

#     return embeddings