from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np

def getDataset(numimages, startnum):
    dataset = [[], []]
    count = 0 #Current number of images added to dataset
    count2 = 0 #Used for counting images until it gets to where it left off
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        if count < int(.9 * numimages):
            images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder) if isfile(join('C:/SSD_Dataset/Images/Training/Fake/' + folder, f))]
            if (count2 < .9 * startnum):
                count2 += len(images)
                continue
            else:
                for image in images:
                    if (count < .9 * numimages):
                        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + image)
                        imgarr = tf.keras.preprocessing.image.img_to_array(img)
                        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
                        dataset[0].append(imgarr)
                        dataset[1].append(1)
                        count += 1
        else:
            break
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')]
    for folder in folders:
        if count < startnum + numimages:
            images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + folder) if isfile(join('C:/SSD_Dataset/Images/Training/Real/' + folder, f))]
            if (count2 < startnum):
                count2 += len(images)
                continue
            else:
                for image in images:
                    if (count < numimages):
                        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Real/' + folder + '/' + image)
                        imgarr = tf.keras.preprocessing.image.img_to_array(img)
                        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
                        dataset[0].append(imgarr)
                        dataset[1].append(0)
                        count += 1
        else:
            break
    return dataset

def getOneImagePerFolder():
    dataset = [[], []]
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder) if isfile(join('C:/SSD_Dataset/Images/Training/Fake/' + folder, f))]
        if len(images) > 0:
            img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + images[0])
            imgarr = tf.keras.preprocessing.image.img_to_array(img)
            imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
            dataset[0].append(imgarr)
        else:
            continue
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + folder) if isfile(join('C:/SSD_Dataset/Images/Training/Real/' + folder, f))]
        if len(images) > 0:
            img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Real/' + folder + '/' + images[0])
            imgarr = tf.keras.preprocessing.image.img_to_array(img)
            imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
            dataset[1].append(imgarr)
        else:
            continue
    return dataset

def getDataRandomized():
    array = []
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')] 
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder) if isfile(join('C:/SSD_Dataset/Images/Training/Fake/' + folder, f))]
        for image in images:
            array.append(['C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + image, 1])
            
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')]
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + folder) if isfile(join('C:/SSD_Dataset/Images/Training/Real/' + folder, f))]
        for image in images:
            array.append(['C:/SSD_Dataset/Images/Training/Real/' + folder + '/' + image, 0])

    array = np.array(array)
    np.random.shuffle(np.array(array))
    return array

def generateBatch(foldername):
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + foldername)]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Real/' + foldername + '/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
        batch[0].append(imgarr)
        batch[1].append(0)
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')]
    base, identifier = foldername.split('_')
    for folder in folders:
        spl = folder.split('_')
        if (spl[0] == base and spl[2] == identifier):
            images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/')]
            for image in images:
                img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + image)
                imgarr = tf.keras.preprocessing.image.img_to_array(img)
                imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
                batch[0].append(imgarr)
                batch[1].append(1)
    for value in batch[1]:
        if value == 1:
            return batch

def getValidationData():
    batch = [[], []]
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Validation/Real/')]
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Validation/Real/' + folder)]
        for image in images:
            img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Validation/Real/' + folder + '/' + image)
            imgarr = tf.keras.preprocessing.image.img_to_array(img)
            imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
            batch[1].append(imgarr)
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Validation/Fake/')]
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Validation/Fake/' + folder)]
        for image in images:
            img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Validation/Fake/' + folder + '/' + image)
            imgarr = tf.keras.preprocessing.image.img_to_array(img)
            imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
            batch[0].append(imgarr)
    return batch

def getV2ValidationData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/valid/fake')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/V2/valid/fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[1].append(imgarr)
    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/valid/real')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/V2/valid/real/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[0].append(imgarr)
    return batch

def createOneBatch(realfolder, fakefolder):
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + realfolder)]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Real/' + realfolder + '/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
        batch[0].append(imgarr)
        batch[1].append(0)
    images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + fakefolder)]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/Training/Fake/' + fakefolder + '/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
        batch[0].append(imgarr)
        batch[1].append(1)
    return batch

def getV2DataRandomized():
    array = []
    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/train/fake')]
    for image in images:
        array.append(['C:/SSD_Dataset/Images/V2/train/fake/' + image, 1])
            
    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/train/real')]
    for image in images:
        array.append(['C:/SSD_Dataset/Images/V2/train/real/' + image, 0])

    array = np.array(array)
    np.random.shuffle(array)
    return array