import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from getData import getDataset, getOneImagePerFolder, getDataRandomized, generateBatch, getValidationData, createOneBatch
from os import listdir
def defineModel():
    inputs_disc = x = tf.keras.Input(shape=(192, 256, 3,))
    #x = layers.experimental.preprocessing.RandomFlip()(x, training=True)
    #x = layers.experimental.preprocessing.RandomRotation((-1,1))(x, training=True)
    #x = layers.experimental.preprocessing.RandomTranslation((-0.2,0.2),(-0.2,0.2))(x, training=True)
    #x = layers.experimental.preprocessing.RandomZoom((-0.1,0))(x, training=True)
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

#Generates batches consisting of real images and each set of fake images that is made from it
def trainModel(model):
    realfolders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')]
    for folder in realfolders:
        batch = generateBatch(folder)
        if batch is not None: # and len(batch[1]) > 150
            X = np.array(batch[0])
            y = np.array(batch[1]).astype(np.float)
            ratio = findRatio(y)
            print(str(len(y)) + ', ' + str(ratio))
            model.fit(X, y, batch_size = len(X), epochs=3, class_weight={1:.3 * ratio, 0:(1-(.3 * ratio))}, shuffle=True)
    model.save('Deepfake_Detector_Model_BigBatches_NoExperimentalLayers.h5')
    return model

#TO DO: Write algorithm that gets data in the following way:
#   For every real folder:
#       For every fake folder corresponding to that real folder:
#           Generate a batch containing the real images and the fake images from their respective folders
#DONE

#Data format:
# Real: idx_001
# Fake: idx_idy_001
# def trainModel(model):
#     realfolders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')]
#     for realfolder in realfolders:
#         split = realfolder.split('_')
#         fakefolders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/') if (f.split('_')[0] == split[0] and f.split('_')[2] == split[1])]
#         for fakefolder in fakefolders:
#             batch = createOneBatch(realfolder, fakefolder)
#             if batch is not None: # and len(batch[1]) > 150
#                 X = np.array(batch[0])
#                 y = np.array(batch[1]).astype(np.float)
#                 model.fit(X, y, batch_size = len(X), epochs=2, shuffle=True)
#     model.save('Deepfake_Detector_Model_Batches_Real_Corresponding_Fake_No_Exp_Layers.h5')
#     return model


# def trainModel(model):
#     i = 0
#     dat = getDataRandomized()
#     while (i < 10):
#         batch = dat[np.random.randint(dat.shape[0], size=10000), :]
#         X = getDataFromList(batch[:,0])
#         y = batch[:,1].astype(np.float)
#         ratio = findRatio(y)
#         print(ratio)
#         model.fit(X, y, epochs=10, batch_size=1000,class_weight={1:ratio, 0:(1 - ratio)}, validation_split=.1, validation_batch_size=500, shuffle=True)
#         i += 1
#     model.save('Deepfake_Detector_Model.h5')
#     return model

def loadModel():
    return tf.keras.models.load_model('48-93_Deepfake_Detector_Model_BigBatches_NoExperimentalLayers.h5')

#Mode 0 evaluates with one image from each folder in training data, mode 1 evaluates with separate validation dataset
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

#Mode 0 evaluates with one image from each folder in training data, mode 1 evaluates with separate validation dataset
def evaluateConfusion(model, mode):
    #model.summary()
    if (mode == 0):
        dataset = getOneImagePerFolder()
    elif (mode == 1):
        dataset = getValidationData()
    X1 = np.array(dataset[0])
    X2 = np.array(dataset[1])
    fake_preds = model.predict(X1, batch_size=32)
    real_preds = model.predict(X2, batch_size=32)

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for value in fake_preds:
        if value > .5:
            TP += 1
        else:
            FN += 1

    for value in real_preds:
        if value < .5:
            TN += 1
        else:
            FP += 1
    print("True Positives (Fake): " + str(TP))
    print("False Positives: " + str(FP))
    print("False Negatives: " + str(FN))
    print("True Negatives (Real): " + str(TN))

#evaluateModel(trainModel(defineModel()), 1)
evaluateConfusion(loadModel(), 1)
evaluateModel(loadModel(), 1)

print("Done")