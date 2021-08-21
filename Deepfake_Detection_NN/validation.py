import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 10000 #10,000
image_resize = (256,256) #(256,256)

# Recreate the exact same model, including its weights and the optimizer
cnn = tf.keras.models.load_model('93_Deepfake_Detector_Model_from_train2.h5')

# Show the model architecture
cnn.summary()

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/valid',
                                            target_size = image_resize,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

loss, acc = cnn.evaluate(test_set, verbose=1)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))