from cv2 import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# set video file path of input video with name and extension
path_to_data = './data/Fake_Videos/'
path_to_labeled_data = './data/Fake_Unlabeled_Images/'

onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))] #Get list with all name of files in directory specified

for value in onlyfiles:
    vid = cv2.VideoCapture(join(path_to_data, value))
    name = value[:-4]
    if not os.path.exists(path_to_labeled_data + name): #Create Path to save images to
        os.makedirs(path_to_labeled_data + name) 
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
            path = path_to_labeled_data + name + '/' + name + '_' + str(imgcount) + '.jpg' #Path to save image to with image name
            cv2.imwrite(path, frame)
            print(name + '/' + name + '_' + str(imgcount) + '.jpg')
            imgcount += 1

        # next frame
        index += 1
    

