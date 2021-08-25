from os import path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from getData import read_csv, getImages_cropped
from model import get_cnn_localization_model

#params
batch_size = 4
image_resize = (100,100) #(256,256)
training_path_to_imgs= 'C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/train'
testing_path_to_imgs =  'C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/test'


#model = get_cnn_localization_model((image_resize[0],image_resize[1],3))
images_path, labels, regbox = read_csv()
length = len(images_path)
part = int(length/batch_size)

i = 0
while (i<part):
    
    label_batches = None
    image_batches = None
    regbox_batches = None
    #partition
    if i+1 < part:
        label_batches = labels[batch_size * i:batch_size * (i + 1)]
        image_batches = images_path[batch_size * i:batch_size * (i + 1)]
        regbox_batches = regbox[batch_size * i:batch_size * (i + 1)]
    else:
        label_batches = labels[batch_size * i:]
        image_batches = images_path[batch_size * i:]
        regbox_batches = regbox[batch_size * i:]

    print(label_batches)

    i += 1