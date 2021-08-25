import csv
import cv2
from utils.opencv_face_detection import cv2_face_cropper
import numpy as np

def read_csv(csv_path='converted_path.csv'):
    images_path = []
    regbox = []
    labels = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            line = row[0].split(',')
            img_dir = line[0]
            label = int(line[1])
            x = int(line[2])
            y = int(line[3])
            w = int(line[4])
            h = int(line[5])
            images_path.append(img_dir)
            labels.append(label)
            regbox.append((x,y,w,h))
    
    return images_path, labels, regbox

def getImages_cropped(filelist):
    face_cropper = cv2_face_cropper()
    data = []
    for file in filelist:
        faces,_ = face_cropper.getfaces_withCord(file)
        face = faces[0]['img']
        face = cv2.resize(face, (256, 256))
        face = (face)/255.0
        data.append(face)
    return data
    