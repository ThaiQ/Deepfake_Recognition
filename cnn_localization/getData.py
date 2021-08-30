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

def getImages_data_andResizeCord(filelist, cordList, img_size=(100,100)):
    imageData = []
    newcordList = []
    
    i = 0
    while i < len(filelist):
        #load image
        face = cv2.imread(filelist[i])
        (h, w) = face.shape[:2]
        face = cv2.resize(face, (img_size[0], img_size[1]))
        face = (face)/255.0
        imageData.append(face)
        newcordList.append([
            cordList[i][0]/w,
            cordList[i][1]/h,
            cordList[i][2]/w,
            cordList[i][3]/h,
            ])
        i+=1
    return imageData, newcordList

def getImages_data(filelist, img_size=(100,100)):
    imageData = []
    
    i = 0
    while i < len(filelist):
        #load image
        face = cv2.imread(filelist[i])
        face = cv2.resize(face, (img_size[0], img_size[1]))
        face = (face)/255.0
        imageData.append(face)

        i+=1
    return imageData

def reverse_RegBox_size(regbox, ratio):
    newReg = (
        regbox[0]/ratio,
        regbox[1]/ratio,
        regbox[2]/ratio,
        regbox[3]/ratio,
    )
    return newReg
    