import flask
import face_recognition
from flask import Flask, request, jsonify, render_template

from bob_assistant.face.data import FEATURES
from bob_assistant.face.compare import picture_to_bytes, \
    verify_user


app = Flask(__name__)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return '<h1>Hello Netgural!</h1>'


@app.route('/verify', methods=['POST'])
def verify():
    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400

        img_file = request.files.get('file')
        img_name = img_file.filename
        img_file_encoded = face_recognition.face_encodings(img_file)
        return verify_user(img_file_encoded, FEATURES)


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, Nothing at this URL.', 404


@app.errorhandler(500)
def page_not_found(e):
    """Return a custom 500 error."""
    return 'Sorry, unexpected error: {}'.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
