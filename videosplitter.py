from cv2 import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# set video file path of input video with name and extension

onlyfiles = [f for f in listdir('C:/Github/Deepfake_Recognition_SSD/Videos/Real Videos') if isfile(join('C:/Github/Deepfake_Recognition_SSD/Videos/Real Videos/', f))]

for value in onlyfiles:
    vid = cv2.VideoCapture(join('C:/Github/Deepfake_Recognition_SSD/Videos/Real Videos/', value))
    name = value[:-4]
    if not os.path.exists('C:/Github/Deepfake_Recognition_SSD/Unlabeled Images/Real Unlabeled Images/' + name):
        os.makedirs('C:/Github/Deepfake_Recognition_SSD/Unlabeled Images/Real Unlabeled Images/' + name)
    #for frame identity
    print("Working on video " + name)
    index = 0
    imgcount = 0
    while(True):
        # Extract images
        ret, frame = vid.read()
        # end of frames
        if not ret: 
            break
        # Saves images
        if (index % 20 == 0):
            path = 'C:/Github/Deepfake_Recognition_SSD/Unlabeled Images/Real Unlabeled Images/' + name + '/' + name + '_' + str(imgcount) + '.jpg'
            cv2.imwrite(path, frame)
            print(name + '/' + name + '_' + str(imgcount) + '.jpg')
            imgcount += 1

        # next frame
        index += 1
    

