import face_recognition
import numpy as np


def picture_to_bytes(picture, jitters=1):
    picture_to_store = face_recognition.load_image_file(picture)
    face_encoding = face_recognition.face_encodings(
        picture_to_store, num_jitters=jitters)
    return face_encoding[0].tobytes()


def compare_features(unknown_features, known_features, tolerance=0.7):
    result = face_recognition.compare_faces(known_features,
                                            unknown_features,
                                            tolerance=tolerance)

    return result


def verify_user(unknown, known_users):
    for _, val in known_users.items():
        known = val.get('face_features')
        known = np.frombuffer(known)
        try:
            if compare_features(unknown, known):
                return val
            else:
                return {'error': 'unknown user'}
        except ValueError:
            return {'error': 'Use RGB pictures'}
