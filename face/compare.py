from PIL import Image
import numpy as np
from numpy import expand_dims
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
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


def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples)
    # create a vggface model
    model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    # perform prediction
    yhat = model.predict(samples)
    return yhat


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return True
    else:
        return False


def verify_user(unknown, known_users, tolerance=0.5):
    for _, val in known_users.items():
        known = val.get('face_features')
        # known = np.frombuffer(known)
        try:
            if is_match(known, unknown, tolerance=tolerance):
                return val
            else:
                return {'error': 'unknown user'}
        except ValueError:
            return {'error': 'Use RGB pictures'}
