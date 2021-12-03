import os
from os import listdir
from os.path import isfile, join
from cv2 import cv2

avgwidth = 0
avgheight = 0
count = 0
folders = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Fake_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Fake_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Resized_Fake_Images/' + folder, f))]
    for image in images:
        img = cv2.imread('C:/SSD_Dataset/Images/Resized_Fake_Images/' + folder + '/' + image)
        avgwidth += len(img[0])
        avgheight += len(img)
        count += 1

folders = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Real_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder, f))]
    for image in images:
        img = cv2.imread('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder + '/' + image)
        avgwidth += len(img[0])
        avgheight += len(img)
        count += 1

#Average is 184, 255, but do 192, 256
avgwidth = avgwidth / count 
avgheight = avgheight / count
print("Average Width: " + str(avgwidth) + '\nAverage Height: ' + str(avgheight))