import os
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

# Create an authenticated FaceClient.
#fakeimages = [f for f in listdir('C:/SSD_Dataset/Fake_Unlabeled_Images/')] #Get list with all name of files in fake images directory
resolutionNumbers = dict()
folders = [f for f in listdir('C:/SSD_Dataset/Real_Labeled_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Real_Labeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Real_Labeled_Images/' + folder, f))]
    for image in images:
        tree = ET.parse('C:/SSD_Dataset/Real_Labeled_Images/' + folder + '/' + image)
        root = tree.getroot()
        if (root[4][0].text + ', ' + root[4][1].text) not in resolutionNumbers:
            resolutionNumbers[root[4][0].text + ', ' + root[4][1].text] = 1
        else:
            resolutionNumbers[root[4][0].text + ', ' + root[4][1].text] += 1

folders = [f for f in listdir('C:/SSD_Dataset/Fake_Labeled_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Fake_Labeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Fake_Labeled_Images/' + folder, f))]
    for image in images:
        tree = ET.parse('C:/SSD_Dataset/Fake_Labeled_Images/' + folder + '/' + image)
        root = tree.getroot()
        if (root[4][0].text + ', ' + root[4][1].text) not in resolutionNumbers:
            resolutionNumbers[root[4][0].text + ', ' + root[4][1].text] = 1
        else:
            resolutionNumbers[root[4][0].text + ', ' + root[4][1].text] += 1

with open("resolutionnumbers.txt", 'w+') as file:
    for key in resolutionNumbers:
        file.write(key + ': ' + str(resolutionNumbers[key]) + '\n')
print("Done")
