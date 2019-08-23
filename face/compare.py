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


def extract_face(file_object, required_size=(224, 224)):
    # load image from file
    if isinstance(file_object, np.ndarray):
        pixels = file_object
    else:
        pixels = plt.imread(file_object)
    # create the detector, using default weights
    detector = MTCNN()

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


def is_match(known_embedding, candidate_embedding, tolerance=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding.ravel(), candidate_embedding.ravel())
    if score <= tolerance:
        return True
    else:
        return False


def verify_user(unknown, known_users, tolerance=0.5):
    for _, val in known_users.items():
        known = val.get('face_features')
        known = np.frombuffer(known, dtype='float32').reshape((1, 7, 7, 512))
        try:
            if is_match(known, unknown, tolerance=tolerance):
                return {'name': val.get('name'),
                        'surname': val.get('surname'),
                        'email': val.get('email')}
        except ValueError:
            return {'error': 'Use RGB pictures'}
    return {'error': 'unknown user'}
