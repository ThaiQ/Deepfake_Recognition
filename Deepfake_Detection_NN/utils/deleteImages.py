import os
from os import listdir
from os.path import isfile, join

# Create an authenticated FaceClient.
#fakeimages = [f for f in listdir('C:/SSD_Dataset/Fake_Unlabeled_Images/')] #Get list with all name of files in fake images directory
images = [f for f in listdir('C:/SSD_Dataset/Real_Labeled_Images/')] #Get list with all name of files in directory specified
for folder in images:
    images = [f for f in listdir('C:/SSD_Dataset/Real_Labeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Real_Labeled_Images/' + folder, f))]
    for image in images:
        if image.endswith(".jpg"):
            os.remove(os.path.join('C:/SSD_Dataset/Real_Labeled_Images/' + folder + '/', image))