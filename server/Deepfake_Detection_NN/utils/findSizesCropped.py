import os
from os import listdir
from os.path import isfile, join
from cv2 import cv2

resolutionNumbers = dict()
folders = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Fake_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Fake_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Resized_Fake_Images/' + folder, f))]
    for image in images:
        img = cv2.imread('C:/SSD_Dataset/Images/Resized_Fake_Images/' + folder + '/' + image)
        size = str(len(img[0])) + ', ' + str(len(img))
        if size not in resolutionNumbers:
            resolutionNumbers[size] = 1
        else:
            resolutionNumbers[size] += 1

folders = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Real_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder, f))]
    for image in images:
        img = cv2.imread('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder + '/' + image)
        size = str(len(img[0])) + ', ' + str(len(img))
        if size not in resolutionNumbers:
            resolutionNumbers[size] = 1
        else:
            resolutionNumbers[size] += 1

with open("resnum.txt", 'w+') as file:
    for key in resolutionNumbers:
        file.write(key + ': ' + str(resolutionNumbers[key]) + '\n')
    file.close()
print("Done")
