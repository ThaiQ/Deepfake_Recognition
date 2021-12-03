import os
from os import listdir
from os.path import isfile, join
from cv2 import cv2
import xml.etree.ElementTree as ET

folders = [f for f in listdir('C:/SSD_Dataset/Images/Real_Unlabeled_Images/') if f not in listdir('C:/SSD_Dataset/Images/Resized_Real_Images/')] #Get list with all name of files in directory specified
xmlfolders = [f for f in listdir('C:/SSD_Dataset/Images/Real_Labeled_Images/')]
for folder in folders:
    print(folder)
    if not os.path.exists("C:/SSD_Dataset/Images/Resized_Real_Images/" + folder): #Create Path to save images to
        os.makedirs("C:/SSD_Dataset/Images/Resized_Real_Images/" + folder)
    if folder not in xmlfolders:
        continue 
    images = [f for f in listdir('C:/SSD_Dataset/Images/Real_Unlabeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Images/Real_Unlabeled_Images/' + folder, f))]
    for image in images:
        try: 
            tree = ET.parse('C:/SSD_Dataset/Images/Real_Labeled_Images/' + folder + '/' + image[:-4] + '.xml').getroot()
        except:
            continue
        bndbox = [int(tree[6][4][0].text), int(tree[6][4][1].text), int(tree[6][4][2].text), int(tree[6][4][3].text)]
        if bndbox[0] < 3 or bndbox[1] < 3 or bndbox[2] > int(tree[4][0].text) - 1 or bndbox[3] > int(tree[4][1].text) - 1:
            continue
        img = cv2.imread('C:/SSD_Dataset/Images/Real_Unlabeled_Images/' + folder + '/' + image)
        hmod = int(.2 * (bndbox[3] - bndbox[1]))
        wmod = int(.2 * (bndbox[2] - bndbox[0]))
        if (bndbox[1] - hmod < 5):
            img = img[0:bndbox[3] + hmod, bndbox[0] - wmod:bndbox[2] + wmod]
        elif (bndbox[1] - wmod < 5):
            img = img[bndbox[1] - hmod:bndbox[3] + hmod, 0:bndbox[2] + wmod]
        else:
            img = img[bndbox[1] - hmod:bndbox[3] + hmod, bndbox[0] - wmod:bndbox[2] + wmod]
        try:
            cv2.imwrite('C:/SSD_Dataset/Images/Resized_Real_Images/' + folder + '/' + image, img)
        except:
            continue

print("Done")