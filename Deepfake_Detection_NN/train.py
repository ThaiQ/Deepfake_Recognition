import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from getData import getDataset, getOneImagePerFolder, getDataRandomized, generateBatch, getValidationData
from os import listdir
def defineModel():
    inputs_disc = x = tf.keras.Input(shape=(192, 256, 3,))
    x = layers.experimental.preprocessing.RandomFlip()(x, training=True)
    x = layers.experimental.preprocessing.RandomRotation((-1,1))(x, training=True)
    x = layers.experimental.preprocessing.RandomTranslation((-0.2,0.2),(-0.2,0.2))(x, training=True)
    x = layers.experimental.preprocessing.RandomZoom((-0.1,0))(x, training=True)
    x = layers.Conv2D(16, (4, 4), (1, 1), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x) #Not needed, but good to have
    x = layers.Conv2D(16, (4, 4), (1, 1), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x) #Not needed, but good to have
    x = layers.Conv2D(32, (4, 4), (1, 1), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x) #Not needed, but good to have
    x = layers.Conv2D(32, (4, 4), (1, 1), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x) #Not needed, but good to have
    x = layers.Conv2D(64, (4, 4), (1, 1), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x, training=True) 
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x) #Not needed, but good to have
    x = layers.Conv2D(64, (2, 2), (1, 1), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x, training=True) 
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x) #Not needed, but good to have
    x = layers.Flatten()(x)
    x = layers.Dense(384, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(96, activation="relu")(x)
    outputs_disc = x = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs_disc, outputs=outputs_disc, name="DeepfakeDetector")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model.summary()
    return model

def getDataFromList(filelist):
    dataset = []
    for file in filelist:
        img = tf.keras.preprocessing.image.load_img(file)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='nearest')
        dataset.append(imgarr)
    return np.array(dataset)

def findRatio(y):
    count = 0
    for value in y:
        if value == 0: 
            count += 1
    return (count/len(y))

# def trainModel(model):
#     realfolders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/')]
#     for folder in realfolders:
#         batch = generateBatch(folder)
#         if batch is not None: # and len(batch[1]) > 150
#             X = np.array(batch[0])
#             y = np.array(batch[1]).astype(np.float)
#             ratio = findRatio(y)
#             print(str(len(y)) + ', ' + str(ratio))
#             model.fit(X, y, batch_size = 32, epochs=5, class_weight={1:ratio, 0:(1-ratio)}, shuffle=True)
#     model.save('Deepfake_Detector_Model.h5')
#     return model

def trainModel(model):
    i = 0
    dat = getDataRandomized()
    while (i < 10):
        batch = dat[np.random.randint(dat.shape[0], size=10000), :]
        X = getDataFromList(batch[:,0])
        y = batch[:,1].astype(np.float)
        ratio = findRatio(y)
        print(ratio)
        model.fit(X, y, epochs=10, batch_size=1000,class_weight={1:ratio, 0:(1 - ratio)}, validation_split=.1, validation_batch_size=500, shuffle=True)
        i += 1
    model.save('Deepfake_Detector_Model.h5')
    return model

def loadModel():
    return tf.keras.models.load_model('Deepfake_Detector_Model_BEST.h5')

def evaluateModel(model, mode):
    #model.summary()
    if (mode == 0):
        dataset = getOneImagePerFolder()
    elif (mode == 1):
        dataset = getValidationData()
    X1 = np.array(dataset[0])
    X2 = np.array(dataset[1])
    fake_preds = model.predict(X1, batch_size=32)
    real_preds = model.predict(X2, batch_size=32)

    count = 0
    for value in fake_preds:
        if value > .5:
            count += 1
    print("Fake Image Accuracy: " + str(float(count / len(fake_preds))))

    count = 0
    for value in real_preds:
        if value < .5:
            count += 1
    print("Real Image Accuracy: " + str(float(count / len(real_preds))))

evaluateModel(trainModel(defineModel()), 1)
#evaluateModel(loadModel(), 0)

print("Done")