import logging
import flask
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import json
import keras

from flask import Flask, request, jsonify, render_template
from face.compare import get_embeddings, verify_user, extract_face
from face.data import KNOWN_USERS

app = Flask(__name__)
from keras_vggface.vggface import VGGFace

import tensorflow as tf
keras.backend.clear_session()
tf.get_default_graph()
MODEL = VGGFace(include_top=False, input_shape=(224, 224, 3))


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return '<h1>Hello Netgural!</h1>'


@app.route('/verify', methods=['GET', 'POST'])
def verify():

    if request.method == 'GET':
        return jsonify({'error': 'only POST requests with file'})

    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400

        img = Image.open(request.files['file'])
        img = np.array(img)
        unknown_face = extract_face(img)
        print('\n\n unknown_face \n', unknown_face, flush=True)
        unknown_features = get_embeddings(unknown_face, MODEL)
        print('\n\n unknown_features \n', unknown_features, flush=True)
        return json.dumps(verify_user(unknown_features, KNOWN_USERS))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8088, debug=True)
