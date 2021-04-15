import os
from os import listdir
from os.path import isfile, join

f = open("trainreal.txt","w+")
folders = [f for f in listdir('C:/SSD_Dataset/Images/Real_Unlabeled_Images/')] #Get list with all name of files in directory specified
for folder in folders:
    images = [f for f in listdir('C:/SSD_Dataset/Images/Real_Unlabeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Real_Unlabeled_Images/' + folder, f))]
    for image in images:
        f.write(image[:-4] + "\n")
f.close()