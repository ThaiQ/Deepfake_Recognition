import os
from flask import Flask , request, jsonify, render_template, url_for, redirect
app = Flask(__name__)

app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def hello_world():
    return {'api': 'Welcome'}

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if  request.method == 'GET':
        return "Get method"
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], uploaded_file.filename))
    return "uploaded"


if __name__ == '__main__':
    app.run(debug=True)