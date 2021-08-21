import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from getData import getValidationData_w_conf
import numpy as np

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

# Recreate the exact same model, including its weights and the optimizer
cnn = tf.keras.models.load_model('93_Deepfake_Detector_Model_from_train2.h5')

# Show the model architecture
cnn.summary()

batches = getValidationData_w_conf()

for real in batches[0]:
    test_image = np.expand_dims(real, axis = 0)
    result = cnn.predict(test_image)
    training_set.class_indices
    print(training_set.class_indices)
    if result[0][0] == 1:
        prediction = 'fake'
    else:
        prediction = 'real'
    print(prediction, result)
