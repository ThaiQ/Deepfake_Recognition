from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from labelFace import label_face
import os
from os import listdir
from os.path import isfile, join

#S3 Endpoint: arn:aws:s3:us-west-1:998691707612:accesspoint/access

# This key will serve all examples in this document.
KEY = "47eb4c612cb8494abb9429e649ac26da"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://imagelabeller.cognitiveservices.azure.com/"

# Create an authenticated FaceClient.
fakeimages = [f for f in listdir('C:/SSD_Dataset/Fake_Unlabeled_Images/')] #Get list with all name of files in fake images directory
#realimages = [f for f in listdir('C:/SSD_Dataset/Real_Unlabeled_Images/')] #Get list with all name of files in directory specified
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
for folder in fakeimages:
    if not os.path.exists("C:/SSD_Dataset/Real_Labeled_Images/" + folder): #Create Path to save images to
        os.makedirs("C:/SSD_Dataset/Real_Labeled_Images/" + folder) 
    images = [f for f in listdir('C:/SSD_Dataset/Fake_Unlabeled_Images/' + folder) if isfile(join('C:/SSD_Dataset/Fake_Unlabeled_Images/' + folder, f))]
    for image in images:
        labeledimage = label_face('http://73.70.9.32:8080/img/Fake_Unlabeled_Images/' + folder + '/' + image, face_client)
        labeledimage.save("C:/SSD_Dataset/Real_Labeled_Images/" + folder + "/" + image)
    print("Folder complete")