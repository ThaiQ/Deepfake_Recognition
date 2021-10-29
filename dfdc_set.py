import os
from glob import glob
import json
import cv2

#define directory paths for fake and real images
fake_path = 'data/test_data/FAKE'
real_path = 'data/test_data/REAL' 

# create directories to store real and fake images if they do not exist
if not os.path.exists(fake_path):
  os.makedirs(fake_path)
if not os.path.exists(real_path):
  os.makedirs(real_path)

os.chdir('data/test_data')

#### FRAME CAPTURE PSEUDO CODE ####
# 1. loop through the .mp4 w/glob
# 2. each video should be identified as real or fake first
# 3. open the video with cv2
# 4. each video is ~10 seconds: assume 30fps -> cut every 75 frames
# 5. each frame cut gets stored respectively depending on the identification

# this metadata file contains the labels that we need to distinguish fake from real
f = open('metadata.json',)
metadata = json.load(f)
for file in glob('*.mp4'):
  # use cv2 to do the snip snip
  capture = cv2.VideoCapture(file)
  os.chdir(str(metadata[file]['label']))
  i = 0
  while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
      break
    if i % 75 == 0:      
      # find the faces in this frame
      grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      face_cascade = cv2.CascadeClassifier('../../haarcascade_frontalFace_default.xml')
      faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)
      for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cropped_frame = frame[y:y + h, x:x, w]
        cv2.imwrite(file + '_frame' + str(i) + '.png', cropped_frame)
    i += 1
  capture.release()
  cv2.destroyAllWindows()
  os.chdir('../')


### IMAGE CROP PSEUDO CODE ###
# 1. loop through the images w/ glob
# 2. detect the faces from the image
# 3. crop and resize them
# 4. resave them