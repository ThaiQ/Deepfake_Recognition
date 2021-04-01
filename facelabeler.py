from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from labelFace import label_face

#S3 Endpoint: arn:aws:s3:us-west-1:998691707612:accesspoint/access

# This key will serve all examples in this document.
KEY = "47eb4c612cb8494abb9429e649ac26da"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://imagelabeller.cognitiveservices.azure.com/"

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
imageurl = 'http://127.0.0.1:8000/img.png'
label_face(imageurl, face_client)