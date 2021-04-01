from cv2 import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# set video file path of input video with name and extension

onlyfiles = [f for f in listdir('C:/SSD_Dataset/Fake Videos/') if isfile(join('C:/SSD_Dataset/Fake Videos/', f))] #Get list with all name of files in directory specified

for value in onlyfiles:
    vid = cv2.VideoCapture(join('C:/SSD_Dataset/Fake Videos/', value))
    name = value[:-4]
    if not os.path.exists('C:/SSD_Dataset/Fake Unlabeled Images/' + name): #Create Path to save images to
        os.makedirs('C:/SSD_Dataset/Fake Unlabeled Images/' + name) 
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
            path = 'C:/SSD_Dataset/Fake Unlabeled Images/' + name + '/' + name + '_' + str(imgcount) + '.jpg' #Path to save image to with image name
            cv2.imwrite(path, frame)
            print(name + '/' + name + '_' + str(imgcount) + '.jpg')
            imgcount += 1

        # next frame
        index += 1
    

