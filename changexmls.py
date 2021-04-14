import xml.etree.ElementTree as ET
from os import listdir
import os
from os.path import isfile, join

xmls = [f for f in listdir('C:/SSD_Dataset/Real_Labeled_Images/')] #Get list with all name of files in directory specified
for folder in xmls:
    images = [f for f in listdir('C:/SSD_Dataset/Real_Labeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Real_Labeled_Images/' + folder, f))]
    for image in images:
        tree = ET.parse('C:/SSD_Dataset/Real_Labeled_Images/' + folder + '/' + image)
        root = tree.getroot()
        temp = root[6][4][1].text
        root[6][4][1].text = root[6][4][3].text
        root[6][4][3].text = temp
        tree.write("C:/SSD_Dataset/Real_Labeled_Images/" + folder + "/" + image[:-4] + ".xml")
        #time.sleep(3)