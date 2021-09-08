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

f = open('metadata.json',)
metadata = json.load(f)
for file in glob('*.mp4'):
  #use cv2 to do the snip snip
  capture = cv2.VideoCapture(file)
  i = 0
  while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
      break
    if i % 75 == 0:
      cv2.imwrite(file + '_frame' + str(i) + '.png', frame)
    i += 1
  capture.release()
  cv2.destroyAllWindows()
  #store the image in 'data/test_data/' + metadata[file]['label']
  print(file + 'stored in: data/test_data/' + metadata[file]['label'])

#### PSEUDO CODE ####
# 1. loop through the .mp4 w/glob
# 2. each video should be identified as real or fake first
# 3. open the video with cv2
# 4. each video is ~10 seconds: assume 30fps -> cut every 75 frames
# 5. each frame cut gets stored respectively depending on the identification
