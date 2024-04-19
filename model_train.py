import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os
import pickle
import numpy as np
import cv2
import face_recognition
from deepface import DeepFace


def detect_and_crop_faces(image_path):
    try:
        # Detect faces in the image using RetinaFace (default)
        try:
            result = DeepFace.extract_faces(image_path, detector_backend='retinaface')
        except:
            print("Error in face detection. no face detected.")
            return None
        # Load the original image
        img = cv2.imread(image_path)
        # Save the cropped face
        for i, face in enumerate(result):
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
            cropped_face = img[y:y+h, x:x+w]
            # saving the cropped face
            cv2.imwrite("cropped_face.jpeg", cropped_face)
            
            
            return cropped_face
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

grouped_pickle_files_folder = "train_dataset_pickle_files"
sub_folders = []

for folder in os.listdir(grouped_pickle_files_folder):
    sub_folders.append(folder)
print(sub_folders)

#test photo
input_photo = "George_w_bush_test_image.jpeg"


def get_face_encodings(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations,num_jitters=1,model='large')
    
    return face_encodings
    





# Load the embeddings and train the model
X = []
y = []
for i, sub_folder in enumerate(sub_folders):
    embeddings_folder = os.path.join(grouped_pickle_files_folder, sub_folder)
    # print(embeddings_folder)
    for embedding_file in os.listdir(embeddings_folder):
        # print(embedding_file)
        with open(os.path.join(embeddings_folder, embedding_file), 'rb') as f:
            embeddings = pickle.load(f)
            try:
                # print(embeddings[0].shape)
                #shape is 1(128,)
                X.append(embeddings[0])
                y.append(i)
            except:
                pass

X = np.array(X)
y = np.array(y)

# store this X and Y
np.save("X.npy", X)
np.save("y.npy", y)

# load the X and Y
# X = np.load("X.npy")
# y = np.load("y.npy")

print(X.shape)
print(y.shape)

# x shape is (no of photos, 128) and y shape is (no of photos,) {no of photos here is 719}

# Create the model if not already created
if not os.path.exists("face_recognition_model2.keras"):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(128,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(sub_folders), activation='softmax'))


    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

    # Save the model
    model.save("face_recognition_model.keras")

else:
    # Load the model
    model = keras.models.load_model("face_recognition_model2.keras")
import matplotlib.pyplot as plt

# # Train the model
# history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

# # Plot training history
# plt.figure(figsize=(12, 6))

# # Plot training & validation accuracy values
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# plt.show()

# # Load the test photo
# image = cv2.imread(input_photo)

# # Detect faces in the test photo
# face_image  = detect_and_crop_faces(input_photo )

# # resize the image
# face_image = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_AREA)

# image = face_image
# # Convert the image to RGB (OpenCV uses BGR by default)
# # image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
# # show the image

# # Get the face encodings

# # face_locations = face_recognition.face_locations(img=image)
# # print(face_locations)
# face_encodings = face_recognition.face_encodings(image, num_jitters=1, model='large')
# print(face_encodings)
# # Reshape the face encodings to match the expected input shape
# # face_encodings = np.array(face_encodings).reshape(-1, 128)

# # Predict the person in the test photo
# predictions = model.predict(face_encodings)
# print(predictions)
# predicted_person_index = np.argmax(predictions)
# predicted_person = sub_folders[predicted_person_index]
# print(f"The person in the photo is {predicted_person}")





