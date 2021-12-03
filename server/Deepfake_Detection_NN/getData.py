from os import listdir
import tensorflow as tf
import numpy as np

#Get the training dataset
def getDeepfakeDatasetRandomized():
    array = []
    images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Train/Fake')]
    for image in images:
        array.append(['C:/SSD_Dataset/Deepfakes/Train/Fake/' + image, 1])
            
    images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Train/Real')]
    for image in images:
        array.append(['C:/SSD_Dataset/Deepfakes/Train/Real/' + image, 0])

    array = np.array(array)
    np.random.shuffle(array)
    return array

#Retrieve the images from the specified file path
def getDataFromList(filelist):
    dataset = [[], []]
    for file in filelist:
        img = tf.keras.preprocessing.image.load_img(file[0])
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = imgarr/255.0
        #imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='nearest')
        dataset[0].append(imgarr)
        dataset[1].append(file[1].astype(np.float))
    return dataset

#Load the test dataset
def getFinalValidationData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Test/Fake')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Deepfakes/Test/Fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[1].append(imgarr)
    images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Test/Real')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Deepfakes/Test/Real/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[0].append(imgarr)
    return batch

def getFinalValidationData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Test/Fake')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Deepfakes/Test/Fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[1].append(imgarr)
    images = [f for f in listdir('C:/SSD_Dataset/Deepfakes/Test/Real')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Deepfakes/Test/Real/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[0].append(imgarr)
    return batch

