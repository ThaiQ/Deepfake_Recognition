import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.opencv_face_detection import display_keras_imageGenerator
 
#params
batch_size = 5 #10,000
image_resize = (256,256) #(256,256)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/train',
                                                 target_size = image_resize,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')
