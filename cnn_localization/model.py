import tensorflow as tf
from tensorflow.keras import layers

def get_cnn_localization_model(image_shape=(100,100,3)):
    inputs = tf.keras.Input(shape=image_shape)

    #preprocessing
    # preprocessing = layers.experimental.preprocessing.RandomRotation(0.2)(inputs, training=True)
    # preprocessing = layers.experimental.preprocessing.RandomTranslation(0.2,0.2)(preprocessing, training=True)
    # preprocessing = layers.experimental.preprocessing.RandomFlip('horizontal')(preprocessing, training=True)

    x = layers.Conv2D(filters = 32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    #classifier
    classifier_head = layers.Dropout(0.3)(x)
    classifier_head = layers.Dense(units=128, activation='relu')(classifier_head)
    classifier_head = layers.Dense(units=64, activation='relu')(classifier_head)
    classifier_head = layers.Dense(units=1, activation='sigmoid')(classifier_head)

    #boudingBoxes
    regbox_head = layers.Dropout(0.3)(x)
    regbox_head = layers.Dense(units=128, activation='relu')(regbox_head)
    regbox_head = layers.Dense(units=64, activation='relu')(regbox_head)
    regbox_head = layers.Dense(units=4, activation='sigmoid')(regbox_head)

    model = tf.keras.Model(inputs=inputs, outputs=[classifier_head,regbox_head])
    model.compile(loss=['binary_crossentropy','mse'], optimizer='adam', metrics=['binary_accuracy'])
    model.summary()
    return model