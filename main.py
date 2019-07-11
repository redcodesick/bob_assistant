import flask
from flask import Flask, request, jsonify
# from flask_restful import Resource, Api

app = Flask(__name__)
# api = Api(app)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return '<h1>Hello Netgural!</h1>'

# class HelloWorld(restful.Resource):
#     def get(self):
#         return {'hello': 'Michal'}


# api.add_resource(HelloWorld, '/upload')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400
        img_file = request.files.get('file')
        img_name = img_name = img_file.filename
        return img_name


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
