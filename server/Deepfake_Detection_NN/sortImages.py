import csv
import shutil
import os
from os import listdir
from os.path import isfile, join

# with open('C:/SSD_Dataset/Deepfakes/metadata.csv', newline='') as data:
#     count = 0
#     real = 1
#     fake = 1
#     reader = csv.reader(data, delimiter=',')
#     for row in reader:
#         filename = row[0][:-4] + '.jpg'
#         if row[3] == 'FAKE':
#             shutil.move("C:/SSD_Dataset/Deepfakes/faces_224/" + filename, "C:/SSD_Dataset/Deepfakes/Fake/" + 'fake_' + str(fake) + '.jpg')
#             fake += 1
#         elif row[3] == 'REAL':
#             shutil.move("C:/SSD_Dataset/Deepfakes/faces_224/" + filename, "C:/SSD_Dataset/Deepfakes/Real/" + 'real_' + str(real) + '.jpg')
#             real += 1

images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Fake/')]
for image in images:
    if image[0:4] == 'real':
        shutil.move("C:/SSD_Dataset/Deepfakes/Fake/" + image, "C:/SSD_Dataset/Deepfakes/Real/" + image)
print("done")

