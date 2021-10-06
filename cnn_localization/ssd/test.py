from os import path
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from getData import read_csv, getImages_data_andResizeCord
from model import vgg16_layer
from utils.opencv_face_detection import cv2_face_cropper

#params
batch_size = 20
partition_size = 400
epoch_per_batch=10
image_resize = (300,300) #(256,256)
training_path_to_csv= './converted_path_train.csv'

model = vgg16_layer((image_resize[0],image_resize[1],3))
images_path, labels, regbox = read_csv(training_path_to_csv)
#randomize data
images_path, labels, regbox = shuffle(images_path, labels, regbox)
length = len(images_path)
part = int(length/partition_size)

i = 0
while (i<part):
    
    label_batches = None
    image_batches = None
    regbox_batches = None
    #partition
    if i+1 < part:
        label_batches = labels[partition_size * i:partition_size * (i + 1)]
        image_batches = images_path[partition_size * i:partition_size * (i + 1)]
        regbox_batches = regbox[partition_size * i:partition_size * (i + 1)]
    else:
        label_batches = labels[partition_size * i:]
        image_batches = images_path[partition_size * i:]
        regbox_batches = regbox[partition_size * i:]

    image_batches, regbox_batches = getImages_data_andResizeCord(image_batches, regbox_batches, image_resize)
    image_batches= np.array(image_batches)
    label_batches = np.array(label_batches)
    regbox_batches = np.array(regbox_batches)
    model.fit(image_batches, [label_batches,regbox_batches], epochs=epoch_per_batch, steps_per_epoch = len(image_batches)//batch_size, batch_size=batch_size)
    
    i += 1

model.save('RCNN.h5')
