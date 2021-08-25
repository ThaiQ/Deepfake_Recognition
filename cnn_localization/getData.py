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

def getImages_cropped(filelist, img_size=(100,100)):
    face_cropper = cv2_face_cropper()
    data = []
    image_size_ratio = []
    for file in filelist:
        faces,_ = face_cropper.getfaces_withCord(file)
        face = faces[0]['img']
        oldSize=face.shape[0]
        face = cv2.resize(face, (img_size[0], img_size[1]))
        image_size_ratio.append(face.shape[0]/oldSize)
        face = (face)/255.0
        data.append(face)
    return data, image_size_ratio

def resize_RegBox(regbox_batches, image_size_ratio):
    newReg_return = []
    for reg, ratio in zip(regbox_batches, image_size_ratio):
        newReg = (
            reg[0]*ratio,
            reg[1]*ratio,
            reg[2]*ratio,
            reg[3]*ratio,
        )
        newReg_return.append(newReg)
    print (newReg_return)
    return newReg_return

def reverse_RegBox_size(regbox, ratio):
    newReg = (
        regbox[0]/ratio,
        regbox[1]/ratio,
        regbox[2]/ratio,
        regbox[3]/ratio,
    )
    return newReg
    