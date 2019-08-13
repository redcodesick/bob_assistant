import logging
import flask
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np


from flask import Flask, request, jsonify, render_template
from face.compare import get_embeddings, verify_user, extract_face
from face.data import KNOWN_USERS

app = Flask(__name__)


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

        unknown_features = get_embeddings(img)

        return verify_user(unknown_features, KNOWN_USERS)


# def get_output(filename):
#     try:
#         with open(filename, 'rb') as output_file:
#             return output_file.read().decode(errors="ignore")
#     except IOError:
#         return None


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8088, debug=True)
