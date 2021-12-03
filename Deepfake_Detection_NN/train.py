import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.training.tracking import base
from getData import *
from models import relu256Model, relu224Model
from statistics import mean
import random

#Find the ratio between different classes in the dataset
def findRatio(y):
    count = 0
    for value in y:
        if value == 0: 
            count += 1
    return (count/len(y))

#Train an ensemble of n models
def trainDeepfakeDetectionEnsemble(n=5):
    for x in range(n):
        model = relu224Model()
        i = 0
        dat = getDeepfakeDatasetRandomized()
        while (i < int((len(dat) / 10000)) + 1):
            eps = random.randint(5,7)
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
            model.fit(X, y, epochs=eps, batch_size=int((len(batch[0]) / 10)), shuffle=True, steps_per_epoch = 10, class_weight={1:(.7 * ratio), 0:1 - (.7 * ratio)})
            i += 1 
        model.save('M' + str(x + 1) + '.h5')
        print("Model saved")

#Load a model
def loadModel():
    return tf.keras.models.load_model('step3_Deepfake_Detector_Model_Combined.h5')

#Evaluate the overall performance of the ensemble
def evaluateEnsemble():
    M1 = tf.keras.models.load_model('M1.h5')
    M2 = tf.keras.models.load_model('M2.h5')
    M3 = tf.keras.models.load_model('M3.h5')
    M4 = tf.keras.models.load_model('M4.h5')
    M5 = tf.keras.models.load_model('M5.h5')
    
    dataset = getFinalValidationData()
    fake = np.array(dataset[1])
    real = np.array(dataset[0])

    fpred1 = M1.predict(fake, batch_size=32)
    rpred1 = M1.predict(real, batch_size=32)
    fpred2 = M2.predict(fake, batch_size=32)
    rpred2 = M2.predict(real, batch_size=32)
    fpred3 = M3.predict(fake, batch_size=32)
    rpred3 = M3.predict(real, batch_size=32)
    fpred4 = M4.predict(fake, batch_size=32)
    rpred4 = M4.predict(real, batch_size=32)
    fpred5 = M5.predict(fake, batch_size=32)
    rpred5 = M5.predict(real, batch_size=32)

    fake_preds = []
    real_preds = []
    print("Validating Ensemble")
    count = 0
    for x in range(len(fpred1)):
        fake_preds.append(round(mean([fpred1[x][0], fpred2[x][0], fpred3[x][0], fpred4[x][0], fpred5[x][0]])))
        if fake_preds[x] > .5:
            count += 1
    print("Fake Image Accuracy: " + str(float(count / len(fpred1))))

    count = 0
    for x in range(len(rpred1)):
        real_preds.append(round(mean([rpred1[x][0], rpred2[x][0], rpred3[x][0], rpred4[x][0], rpred5[x][0]])))
        if real_preds[x] <= .5:
            count += 1
    print("Real Image Accuracy: " + str(float(count / len(rpred1))))

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

trainDeepfakeDetectionEnsemble()
evaluateEnsemble()
print("Done")