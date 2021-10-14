import os
from flask import Flask , request, jsonify, render_template, url_for, redirect, json, flash, send_file
from werkzeug.utils import secure_filename
from temp_manager import Temp_Manager
from utils import path, allowed_file

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = path(dir_path)
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = dir_path

@app.route('/', methods=['GET'])
def status():
    response = app.response_class(
        response=json.dumps("HI"),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/static', methods=['GET'])
def serve(): 
    filename = request.args.get('filename')
    try:
        return send_file(dir_path+'/'+str(filename), mimetype='image/png')
    except:
        return "No file 404."

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            response = app.response_class(
                response=json.dumps("Request has no file."),
                status=400,
                mimetype='application/json'
            )
            return response
        else:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                tmp_manager = Temp_Manager()
                img_name,hash = tmp_manager.process(filename)

                response = app.response_class(
                    response=json.dumps({"OriginFile":img_name,"ProcessedFile":hash}),
                    status=200,
                    mimetype='application/json'
                )

                return response

    response = app.response_class(
                response=json.dumps("Request has no file."),
                status=400,
                mimetype='application/json'
            )
    return response


if __name__ == '__main__':
    app.run(debug=True)