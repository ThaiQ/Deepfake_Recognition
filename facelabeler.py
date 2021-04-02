from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from labelFace import label_face
import os
from os import listdir
from os.path import isfile, join
import time
import xml.etree.ElementTree as ET
from cv2 import cv2

#S3 Endpoint: arn:aws:s3:us-west-1:998691707612:accesspoint/access

# This key will serve all examples in this document.
KEY = "47eb4c612cb8494abb9429e649ac26da"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://imagelabeller.cognitiveservices.azure.com/"

# Create an authenticated FaceClient.
#fakeimages = [f for f in listdir('C:/SSD_Dataset/Fake_Unlabeled_Images/')] #Get list with all name of files in fake images directory
realimages = [f for f in listdir('C:/SSD_Dataset/Real_Unlabeled_Images/')] #Get list with all name of files in directory specified
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
for folder in realimages:
    if not os.path.exists("C:/SSD_Dataset/Real_Labeled_Images/" + folder): #Create Path to save images to
        os.makedirs("C:/SSD_Dataset/Real_Labeled_Images/" + folder) 
    images = [f for f in listdir('C:/SSD_Dataset/Real_Unlabeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Real_Unlabeled_Images/' + folder, f))]
    for image in images:
        try:
            labeledimage = label_face('http://73.70.9.32:8080/img/Real_Unlabeled_Images/' + folder + '/' + image, face_client)
            labeledimage[0].save("C:/SSD_Dataset/Real_Labeled_Images/" + folder + "/" + image)
            
            tree = ET.parse('xmltemplate.xml')
            root = tree.getroot()
            root[0].text = "C:/SSD_Dataset/Real_Unlabeled_Images/" + folder
            root[1].text = image
            root[2].text = "C:/SSD_Dataset/Real_Unlabeled_Images/" + folder + '/' + image
            imageshape = (cv2.imread("C:/SSD_Dataset/Real_Unlabeled_Images/" + folder + '/' + image)).shape
            root[4][0].text = str(imageshape[1])
            root[4][1].text = str(imageshape[0])
            root[4][2].text = str(imageshape[2])
            root[6][0].text = "real"
            root[6][4][0].text = str(labeledimage[1][0][0])
            root[6][4][1].text = str(labeledimage[1][1][1])
            root[6][4][2].text = str(labeledimage[1][1][0])
            root[6][4][3].text = str(labeledimage[1][0][1])
            tree.write("C:/SSD_Dataset/Real_Labeled_Images/" + folder + "/" + image[:-4] + ".xml")

            f = open("C:/SSD_Dataset/Real_Labeled_Images/" + folder + "/" + image[:-4] + ".txt","w+")
            #print(str(labeledimage[1]))
            f.write(str(labeledimage[1]))
            f.close()
            #time.sleep(3)
        except Exception as inst:
            print(type(inst))
            print("Encountered an image without a face: " + image)