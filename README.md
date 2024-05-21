# AI Face Tagger

AI Face Tagger is an advanced facial recognition tool designed to streamline the process of finding specific photos from large collections, such as those from a conference or a wedding. Instead of manually sifting through thousands of images, you can use AI Face Tagger to quickly locate photos containing specific individuals or combinations of people.

## Features

- **Single Person Search**: Upload a photo of yourself, and the algorithm will return all photos containing your face from the collection.
- **Multiple Persons Search**: Upload photos of multiple people (e.g., you and your spouse), and the algorithm will return only those photos where all uploaded faces are present together.
- **User-Friendly Interface**: A simple and intuitive web interface built with HTML and Flask.

## Technologies Used

- **Machine Learning Algorithms**: Core of the facial recognition process.
- **DeepFace**: Used for facial recognition and generating .pkl files.
- **Python**: The primary programming language for implementing the algorithms and backend.
- **HTML and Flask**: For creating the basic user interface.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Flask
- DeepFace
- face_recognition
- keras

## Usage

1. **Upload a Reference Photo**: Upload a clear photo of yourself or the individuals you want to find.
2. **Process the Photos**: The application will run the facial recognition algorithm on the entire collection.
3. **View Results**: The application will display all photos where the specified individuals are present.
4. **Download**: Download the desired photos.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) for providing the facial recognition library.
- Flask for the web framework.
- All contributors and supporters of the project.

---

Feel free to reach out for any questions or feedback!
