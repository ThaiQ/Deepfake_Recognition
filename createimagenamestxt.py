import os
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

f = open("fakeimages.txt", 'w')
# Create an authenticated FaceClient.
#fakeimages = [f for f in listdir('C:/SSD_Dataset/Fake_Unlabeled_Images/')] #Get list with all name of files in fake images directory
resolutionNumbers = dict()
folders = [f for f in listdir('C:/SSD_Dataset/Images/Fake_Unlabeled_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Images/Fake_Unlabeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Fake_Unlabeled_Images/' + folder, f))]
    for image in images:
        f.write(folder + '/' + image[:-4] + '\n')
f.close()