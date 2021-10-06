import tensorflow as tf
from tensorflow.keras import layers

def vgg16_layer(image_shape=(300,300,3)):
    #300x300x3
    inputs = tf.keras.Input(shape=image_shape)
    #300x300x64
    x = layers.Conv2D(filters = 64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters = 64, kernel_size=3, padding='same', activation='relu')(x)
    #150x150x128
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters = 128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 128, kernel_size=3, padding='same', activation='relu')(x)
    #75x75x256
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters = 256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 256, kernel_size=3, padding='same', activation='relu')(x)
    #38x38x512
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)
    return x

def ssd(image_shape=(300,300,3)):
    #conv4 - 38x38x512
    x = vgg16_layer(image_shape)
    #classifier
    classifier_headx38 = layers.Flatten()(x)
    classifier_headx38 = layers.Dense(units=252, activation='sigmoid')(classifier_headx38)

    #conv5
    x = layers.Conv2D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)



