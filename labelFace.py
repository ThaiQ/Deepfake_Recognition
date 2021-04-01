import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person


def label_face(imageurl, face_client):

    # Detect a face in an image that contains a single face
    single_face_image_url = imageurl
    single_image_name = os.path.basename(single_face_image_url)
    # We use detection model 3 to get better performance.
    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_03')
    if not detected_faces:
        raise Exception('No face detected from image {}'.format(single_image_name))

    # Convert width height to a point in a rectangle
    def getRectangle(faceDictionary):
        rect = faceDictionary.face_rectangle
        left = rect.left
        top = rect.top
        right = left + rect.width
        bottom = top + rect.height
        
        return ((left, top), (right, bottom))


    # Download the image from the url
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

    # For each face returned use the face rectangle and draw a red box.
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')
    return img