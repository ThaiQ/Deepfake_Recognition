import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.training.tracking import base
from getData import *
from models import sigmoidModel, reluModel, relu256Model
from os import listdir

def findRatio(y):
    count = 0
    for value in y:
        if value == 0: 
            count += 1
    return (count/len(y))

#Generates batches consisting of real images and each set of fake images that is made from it
# def trainModel(model):
#     realfolders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')]
#     for folder in realfolders:
#         batch = generateBatch(folder)
#         if batch is not None: # and len(batch[1]) > 150
#             X = np.array(batch[0])
#             y = np.array(batch[1]).astype(np.float)
#             ratio = findRatio(y)
#             print(str(len(y)) + ', ' + str(ratio))
#             model.fit(X, y, batch_size = len(X), epochs=5, class_weight={1:.55 * ratio, 0:(1-(.55 * ratio))}, shuffle=True)
#             #model.fit(X, y, batch_size = len(X), epochs=3, class_weight={1:.045, 0:.955}, shuffle=True)
#     model.save('Deepfake_Detector_Model_BigBatches_NoExperimentalLayers.h5')
#     return model

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

#BEST so far
# def trainModel(model):
#     i = 0
#     dat = getDataRandomized()
#     while (i < 10):
#         batch = dat[np.random.randint(dat.shape[0], size=10000), :]
#         X = getDataFromList(batch[:,0])
#         y = batch[:,1].astype(np.float)
#         ratio = findRatio(y)
#         print(ratio)
#         model.fit(X, y, epochs=10, batch_size=1000,class_weight={1:.90 * ratio, 0:(1 - (.90 * ratio))}, validation_split=.1, validation_batch_size=500, shuffle=True, steps_per_epoch = 5)
#         i += 1
#     model.save('Deepfake_Detector_Model.h5')
#     return model

def trainV1(model):
    dat = getDataRandomized() #Loads the file locations of every image in the dataset
    i = 0
    while (i < 10):
        print("Batch number " + str(i + 1))
        batch = getDataFromListCropped(dat[10000 * i:10000 * (i + 1)]) #Selects 10000 images
        print(len(batch[0]))
        X = np.array(batch[0])
        y = np.array(batch[1])
        ratio = findRatio(y)
        print(ratio)
        model.fit(X, y, epochs=3, batch_size=1000, shuffle=True, steps_per_epoch = 10, class_weight={1:(1.5 * ratio), 0:(1 - (1.5 * ratio))})
        i += 1 
    #model.save('Deepfake_Detector_Model_Crops.h5')
    return model

def trainV2(model):
    i = 0
    dat = getV2DataRandomized() #Loads the file locations of every image in the dataset
    while (i < 10):
        print("Batch number " + str(i + 1))
        batch = getDataFromListCropped(dat[10000 * i:10000 * (i + 1)]) #Selects 10000 images
        print(len(batch[0]))
        X = np.array(batch[0])
        y = np.array(batch[1])
        ratio = findRatio(y)
        print(ratio)
        model.fit(X, y, epochs=10, batch_size=1000, shuffle=True, steps_per_epoch = 10, class_weight={1:(1.1 * ratio), 0:(1 - (1.1 * ratio))})
        i += 1 
    #model.save('Deepfake_Detector_Model_Crops.h5')
    return model

def trainCombined(model):
    i = 0
    dat = getCombinedDatasetRandomized()
    while (i < int((len(dat) / 10000)) + 1):
        print("Batch number " + str(i + 1))
        if (i < int((len(dat) / 10000))):
            batch = getDataFromList(dat[10000 * i:10000 * (i + 1)]) #Selects 10000 images
        else:
            batch = getDataFromList(dat[10000 * i:len(dat)]) #Selects 10000 images
        print(len(batch[0]))
        X = np.array(batch[0])
        y = np.array(batch[1])
        ratio = findRatio(y)
        print(ratio)
        model.fit(X, y, epochs=1, batch_size=1000, shuffle=True, steps_per_epoch = 10, class_weight={1:.80, 0:.20})
        i += 1 
    model.save('step4_Deepfake_Detector_Model_Combined.h5')
    return model

def loadModel():
    return tf.keras.models.load_model('step3_Deepfake_Detector_Model_Combined.h5')

#Mode 0 evaluates with one image from each folder in training data, mode 1 evaluates with separate validation dataset
def evaluateModel(model, mode):
    #model.summary()
    if (mode == 0):
        dataset = getOneImagePerFolder()
        print("Validating with V1 dataset")
    elif (mode == 1):
        dataset = getValidationData()
        print("Validating with V1 dataset")
    elif(mode == 2):
        dataset = getV2ValidationData()
        print("Validating with V2 dataset")
    elif(mode == 3):
        dataset = getV2TestData()
        print("Validating with V2 dataset")
    elif(mode == 4):
        dataset = getV3ValidationData()
        print("Validating with V3 dataset")
    X1 = np.array(dataset[1])
    X2 = np.array(dataset[0])
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
    return model

#evaluateModel(trainV2(relu256Model()), 5)
#evaluateModel(evaluateModel(trainV1(trainV2(relu256Model())), 1), 2)
#evaluateModel(evaluateModel(loadModel(), 2), 1)
evaluateModel(evaluateModel(trainCombined(loadModel()), 2), 1)
#evaluateModel(evaluateModel(trainV1(loadModel()), 2), 1)

#cropImages()

print("Done")