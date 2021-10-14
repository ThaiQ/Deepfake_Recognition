import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def path(dir_path):

    OS = os.name
    if OS == "nt":
        dir_path = dir_path + "\\uploads"
    else:
        dir_path = dir_path + "/uploads"

    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(dir_path)

    return dir_path
