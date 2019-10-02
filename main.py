import logging
import flask
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import json
import keras
from flask_sqlalchemy import SQLAlchemy

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'postgres+psycopg2://postgres:l8ykgN8GyHrzj25G@/face_recognition?host=/cloudsql/third-fold-242908:europe-west3:face-recognition'
db = SQLAlchemy(app)

from face.compare import get_embeddings, verify_user, extract_face

from keras_vggface.vggface import VGGFace

import tensorflow as tf
keras.backend.clear_session()
tf.get_default_graph()
MODEL = VGGFace(include_top=False, input_shape=(224, 224, 3))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String)
    face_embeddings = db.Column(db.LargeBinary)

# db.create_all()

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
        return json.dumps(verify_user(unknown_features, User.query.all()))

@app.route('/post_photo', methods=['POST'])
def post_photo():
    if not 'file' in request.files:
        return jsonify({'error': 'no file'}), 400

    if not 'email' in request.args:
        return jsonify({'error': 'no email'}), 400

    img = Image.open(request.files['file'])
    img = np.array(img)
    try:
        unknown_face = extract_face(img)
    except ValueError:
        return jsonify({'error': 'no face'}), 500
    unknown_features = get_embeddings(unknown_face, MODEL)
    user = next(iter(User.query.filter_by(email=request.args['email']).all()), None)
    if user:
        user.face_embeddings = unknown_features
    else:
        user = User(email=request.args['email'], face_embeddings=unknown_features.tobytes())
    db.session.add(user)
    db.session.commit()
    return jsonify({"success": user.email})

@app.route('/destroy', methods=['DELETE'])
def destroy():
    email = request.args['email']
    user = User.query.filter_by(email=email).first_or_404()
    db.session.delete(user)
    db.session.commit()
    return jsonify({"success": user.email})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8088, debug=True)
