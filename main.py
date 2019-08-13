import flask
import matplotlib.pyplot as plt
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from face.compare import get_embeddings, verify_user


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

        img_file = BytesIO(request.files.get('file').read())
        # img_name = img_file.filename
        face = get_embeddings(img_file)

        return jsonify(face)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8088, debug=True)
