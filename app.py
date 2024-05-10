# #getting a new image and finding the closest match
# import cosine_sim
# import pickle
# import os
from cosine_sim import cosine_similarity, make_pickle_files
# output_folder = 'data_new/new_dbscann_jitter_1new_eps0.34_large_final_new'

# # for two person we will have 2 images.. we will return the image path where both the images(person) are present
# # we will return the image path where both the images(person) are present, using prefix logic
# # we will have two images
# # person1 = 'data_new/darshan_test_image.jpeg'
# # person2 = 'data_new/nishith_sir_test.jpeg'
# person1_embedding = make_pickle_files(person1)[0]
# person2_embedding = make_pickle_files(person2)[0]


# # first getting the closest match for the perosn1
# person1_folder_path = cosine_sim.find_closest_match_majority(person1, output_folder, 0.9)
# # now getting the closest match for the person2
# person2_folder_path = cosine_sim.find_closest_match_majority(person2, output_folder, 0.9)

# print(person1_folder_path)
# print(person2_folder_path)

# # now checking if the person2 is present in the person1 folder
# # getting the prefix before the first '.'
# cropped_face_folder = 'data_new/cropped_faces_new'

# images_with_both_persons = []
# original_images_folder = 'data_new/train_dataset_new'
# for image in os.listdir(person1_folder_path):
#     postfix = image.split('.')[0]
#     person1_cropped_folder_path = os.path.join(cropped_face_folder, postfix)
#     # now checking if the person2 is present in the person1 folder
#     for embedding in os.listdir(person1_cropped_folder_path):
#         if embedding.endswith('.pkl'):
#             with open(os.path.join(person1_cropped_folder_path, embedding), 'rb') as f:
#                 embeddings = pickle.load(f)
#                 embedding_data = embeddings[0]
#                 # checking cosine similarity
#                 if cosine_similarity(embedding_data, person2_embedding) > 0.95:
#                     # appending the original image path where both the persons are present after getting the name of the photo prefixing from '.'
#                     images_with_both_persons.append(os.path.join(original_images_folder, embedding.split('.')[0] + '.jpeg'))

# # now for person 2 

# for image in os.listdir(person2_folder_path):
#     postfix = image.split('.')[0]
#     person2_cropped_folder_path = os.path.join(cropped_face_folder, postfix)
#     # now checking if the person1 is present in the person1 folder
#     for embedding in os.listdir(person2_cropped_folder_path):
#         if embedding.endswith('.pkl'):
#             with open(os.path.join(person2_cropped_folder_path, embedding), 'rb') as f:
#                 embeddings = pickle.load(f)
#                 embedding_data = embeddings[0]
#                 # checking cosine similarity
#                 if cosine_similarity(embedding_data, person1_embedding) > 0.95:
#                     # images_with_both_persons.append(os.path.join(person2_cropped_folder_path, embedding))
#                     images_with_both_persons.append(os.path.join(original_images_folder, embedding.split('.')[0] + '.jpeg'))
                    
 
 
# # print(images_with_both_persons)
from flask import Flask, render_template, request, jsonify
import os
import cosine_sim
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@cross_origin
@app.route('/get_matching_images', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded images
        person1_image = request.files.get('person1')
        person2_image = request.files.get('person2')

        # Save the images to the server
        person1_path = os.path.join('uploads', person1_image.filename)
        person1_image.save(person1_path)

        # Process the person1 image
        person1_embedding = cosine_sim.make_pickle_files(person1_path)[0]
        output_folder = 'data_new/new_dbscann_jitter_1new_eps0.34_large_final_new'
        person1_folder_path = cosine_sim.find_closest_match_majority(person1_path, output_folder, 0.95)
        print(f'Person1 folder path: {person1_folder_path}')
        if(not person1_folder_path):
            return jsonify({'images': []})
        image_paths_person_1 = []
        # vraj_darshan.1.jpg , name will be like this we have to extract vraj_darshan
        for image in os.listdir(person1_folder_path):
            image_name = image.split('.')[0]
            image_name += '.jpg'
            image_paths_person_1.append(os.path.join('data_new/train_dataset_new', image_name))
        images_with_person1 = list(set(image_paths_person_1))
        # images_with_person1 = [ os.path.join('data_new/train_dataset_new', image) for image in os.listdir(person1_folder_path) if image.endswith('.jpeg') or image.endswith('.jpg') or image.endswith('.png')]

        images_with_both_persons = []

        # Check if person2 image is provided
        if person2_image:
            person2_path = os.path.join('uploads', person2_image.filename)
            person2_image.save(person2_path)

            # Process the person2 image
            person2_embedding = cosine_sim.make_pickle_files(person2_path)[0]
            person2_folder_path = cosine_sim.find_closest_match_majority(person2_path, output_folder, 0.9)

            cropped_face_folder = 'data_new/cropped_faces_new'
            original_images_folder = 'data_new/train_dataset_new'

            for image in os.listdir(person1_folder_path):
                postfix = image.split('.')[0]
                person1_cropped_folder_path = os.path.join(cropped_face_folder, postfix)
                for embedding in os.listdir(person1_cropped_folder_path):
                    if embedding.endswith('.pkl'):
                        with open(os.path.join(person1_cropped_folder_path, embedding), 'rb') as f:
                            embeddings = pickle.load(f)
                            embedding_data = embeddings[0]
                            if cosine_similarity(embedding_data, person2_embedding) > 0.95:
                                images_with_both_persons.append(os.path.join(original_images_folder, embedding.split('.')[0] + '.jpg'))

            for image in os.listdir(person2_folder_path):
                postfix = image.split('.')[0]
                person2_cropped_folder_path = os.path.join(cropped_face_folder, postfix)
                for embedding in os.listdir(person2_cropped_folder_path):
                    if embedding.endswith('.pkl'):
                        with open(os.path.join(person2_cropped_folder_path, embedding), 'rb') as f:
                            embeddings = pickle.load(f)
                            embedding_data = embeddings[0]
                            if cosine_similarity(embedding_data, person1_embedding) > 0.95:
                                images_with_both_persons.append(os.path.join(original_images_folder, embedding.split('.')[0] + '.jpg'))

            print(images_with_both_persons)
            images_with_both_persons = list(set(images_with_both_persons))
            return jsonify({'images': images_with_both_persons})

        else:
            
            print(images_with_person1)
            return jsonify({'images': images_with_person1})

    # return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)