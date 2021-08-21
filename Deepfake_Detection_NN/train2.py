import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.opencv_face_detection import display_keras_imageGenerator
 
#params
batch_size = 10000 #10,000
image_resize = (256,256) #(256,256)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/train',
                                                 target_size = image_resize,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/test',
                                            target_size = image_resize,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation='relu', input_shape=[image_resize[0], image_resize[1], 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flatten
cnn.add(tf.keras.layers.Flatten())

#fully connected
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 100)

cnn.save('Deepfake_Detector_Model_from_train2.h5')