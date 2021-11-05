from cv2 import cv2
import numpy as np
import os
import imageio
from os import listdir
from os.path import isfile, join
from glob import glob
from Deepfake_Detection_NN.utils.predict import predict_visual

# set video file path of input video with name and extension
# path_to_data = './data/Fake_Videos/'
# path_to_labeled_data = './data/Fake_Unlabeled_Images/'

# onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))] #Get list with all name of files in directory specified

# for value in onlyfiles:
#     vid = cv2.VideoCapture(join(path_to_data, value))
#     name = value[:-4]
#     if not os.path.exists(path_to_labeled_data + name): #Create Path to save images to
#         os.makedirs(path_to_labeled_data + name) 
#     #for frame identity
#     print("Working on video " + name)
#     index = 0
#     imgcount = 0
#     while(True):
#         # Extract images
#         ret, frame = vid.read()
#         # end of frames
#         if not ret: 
#             break
#         # Saves images
#         if (index % 20 == 0):
#             path = path_to_labeled_data + name + '/' + name + '_' + str(imgcount) + '.jpg' #Path to save image to with image name
#             cv2.imwrite(path, frame)
#             print(name + '/' + name + '_' + str(imgcount) + '.jpg')
#             imgcount += 1

#         # next frame
#         index += 1
    
def split_vid(vid_path):
    path = './uploads/video_frames'
    if not os.path.exists(path):
      os.makedirs(path)
    capture = cv2.VideoCapture(vid_path)
    i = 0
    while(capture.isOpened()):
        ret, frame = capture.read()
        if not ret:
            break
        if i % 20 == 0:
            cv2.imwrite(path + '/video_frame' + str(i) + '.jpg', frame)
        i += 1
    capture.release()
    cv2.destroyAllWindows()

def stitch_vid():
  models = ['./models/'+f for f in listdir('./models')]
  models = list(filter(lambda f: f[-2:] == 'h5', models))

  img_siz = (224,224)
  images = []
  save_path = './uploads/predicted_frames/'
  if not os.path.exists(save_path):
      os.makedirs(save_path)
  for image in glob('./uploads/video_frames/*.jpg'):
      predict_visual(image_resize_value=img_siz, model_paths=models, path_to_img=[image], save=save_path, draw = True, show = False)
      os.remove(image)
  for image in glob('./uploads/predicted_frames/*.png'):
      images.append(imageio.imread(image))
      os.remove(image)
  gif_path = './uploads/'
  imageio.mimsave(gif_path + 'prediction.gif', images) #name can be chagned later
