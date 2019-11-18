import json
import logging
import os
import hashlib
from io import BytesIO

import flask
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot as plt
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from face.compare import extract_face, get_embeddings, verify_user


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ['DATABASE_URL']
db = SQLAlchemy(app)
BUCKET_NAME = "pictures_bucket"
keras.backend.clear_session()
tf.get_default_graph()
MODEL = VGGFace(include_top=False, input_shape=(224, 224, 3))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String)
    face_embeddings = db.Column(db.LargeBinary)
    photos = db.relationship("Photo", backref='user')
    face_recognition_enabled = db.Column(db.Boolean, default=True)

class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    photo_link = db.Column(db.String)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return '<h1>Hello Netgural!</h1>'


@app.route('/toggle_face_recognition', methods=['POST'])
def toggle_face_recognition():

    if not 'email' in request.args:
        return jsonify({'error': 'No email in params'}), 400
    try:
        user = User.query.filter_by(email=request.args['email']).one()
    except NoResultFound:
        return jsonify({'error': 'No user'}), 404
    user.face_recognition_enabled = not user.face_recognition_enabled
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify(f'Success switching face recognition to {user.face_recognition_enabled}')
    except SQLAlchemyError:
        return jsonify({'error': 'Cannot toggle face recognition'}), 500


@app.route('/check_face_recognition', methods=['GET'])
def check_face_recognition():

    if not 'email' in request.args:
        return jsonify({'error': 'No email in params'}), 400
    try:
        user = User.query.filter_by(email=request.args['email']).one()
    except NoResultFound:
        return jsonify({'error': 'No user'}), 404
    return jsonify(f'Face recognition is set to {user.face_recognition_enabled}')

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
        return json.dumps(verify_user(unknown_features, User.query.filter_by(face_recognition_enabled = True).all()))

@app.route('/post_photo', methods=['POST'])
def post_photo():

    def upload_blob(bucket_name, source_file, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(source_file)

    if not 'file' in request.files:
        return jsonify({'error': 'No file attached'}), 400

    if not 'email' in request.args:
        return jsonify({'error': 'No email in params'}), 400

    login = request.args.get('login', default='false')
    photo_path = 'login' if login != 'false' else 'register'
    img = Image.open(request.files['file'])
    img = np.array(img)
    try:
        unknown_face = extract_face(img)
    except ValueError:
        return jsonify({'error': 'Couldn\'t find face in picture'}), 500
    photo_name = hashlib.md5(str(img).encode('utf-8')).hexdigest() + '.jpeg'
    photo_path = os.path.join(photo_path, request.args['email'], photo_name)
    request.files['file'].seek(0)
    try:
        upload_blob(BUCKET_NAME, request.files['file'].stream, photo_path)
    except GoogleCloudError:
        return jsonify({'error': 'Couldn\'t save photo in Google Cloud'}), 502
    photo = Photo(photo_link=photo_path)
    #  ML part we will change after collecting the photos
    unknown_features = get_embeddings(unknown_face, MODEL)
    try:
        user = User.query.filter_by(email=request.args['email']).one()
    except NoResultFound:
        user = User(email=request.args['email'])
    user.face_embeddings = unknown_features
    user.photos.append(photo)
    try:
        db.session.add(user)
        db.session.add(photo)
        db.session.commit()
    except SQLAlchemyError:
        return jsonify({'error': 'Couldn\'t save photo in database'}), 500
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
