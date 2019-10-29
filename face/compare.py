from PIL import Image
import keras
import numpy as np
from numpy import expand_dims
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# create the detector, using default weights
DETECTOR = MTCNN()

def extract_face(file_object, required_size=(224, 224)):
    # load image from file
    if isinstance(file_object, np.ndarray):
        pixels = file_object
    else:
        pixels = plt.imread(file_object)

    detector = DETECTOR

    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# extract faces and calculate face embeddings for a list of photo files


def get_embeddings(face, model):
    # prepare the face for the model, e.g. center pixels
    sample = asarray(face, 'float32')
    sample = preprocess_input(sample)
    sample = np.expand_dims(sample, axis=0)

    # perform prediction
    feature_map = model.predict(sample)
    return feature_map


def calculate_similarity(known_embedding, candidate_embedding):
    # calculate distance between embeddings
    return cosine(known_embedding.ravel(), candidate_embedding.ravel())


def verify_user(unknown, users, tolerance=0.5):
    users_list = []
    for user in users:
        known = user.face_embeddings
        known = np.frombuffer(known, dtype='float32').reshape((1, 7, 7, 512))
        try:
            score = calculate_similarity(known, unknown)
            users_list.append((score, user.email))
        except ValueError:
            return {'error': 'Use RGB pictures'}
    users_list.sort(key= lambda _tuple: _tuple[0])
    if users_list[0][0] < 0.45:
        return {'email': users_list[0][1]}
    return {'error': 'unknown user'}
