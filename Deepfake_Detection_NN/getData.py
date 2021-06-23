from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np

def getDataset(numimages, startnum):
    dataset = [[], []]
    count = 0 #Current number of images added to dataset
    count2 = 0 #Used for counting images until it gets to where it left off
    folders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Fake/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        if count < int(.9 * numimages):
            images = [f for f in listdir('D:/SSD_Dataset/Images/Training/Fake/' + folder) if isfile(join('D:/SSD_Dataset/Images/Training/Fake/' + folder, f))]
            if (count2 < .9 * startnum):
                count2 += len(images)
                continue
            else:
                for image in images:
                    if (count < .9 * numimages):
                        img = tf.keras.preprocessing.image.load_img('D:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + image)
                        imgarr = tf.keras.preprocessing.image.img_to_array(img)
                        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
                        dataset[0].append(imgarr)
                        dataset[1].append(1)
                        count += 1
        else:
            break
    folders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/')]
    for folder in folders:
        if count < startnum + numimages:
            images = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/' + folder) if isfile(join('D:/SSD_Dataset/Images/Training/Real/' + folder, f))]
            if (count2 < startnum):
                count2 += len(images)
                continue
            else:
                for image in images:
                    if (count < numimages):
                        img = tf.keras.preprocessing.image.load_img('D:/SSD_Dataset/Images/Training/Real/' + folder + '/' + image)
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
    folders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Fake/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        images = [f for f in listdir('D:/SSD_Dataset/Images/Training/Fake/' + folder) if isfile(join('D:/SSD_Dataset/Images/Training/Fake/' + folder, f))]
        if len(images) > 0:
            img = tf.keras.preprocessing.image.load_img('D:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + images[0])
            imgarr = tf.keras.preprocessing.image.img_to_array(img)
            imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
            dataset[0].append(imgarr)
        else:
            continue
    folders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        images = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/' + folder) if isfile(join('D:/SSD_Dataset/Images/Training/Real/' + folder, f))]
        if len(images) > 0:
            img = tf.keras.preprocessing.image.load_img('D:/SSD_Dataset/Images/Training/Real/' + folder + '/' + images[0])
            imgarr = tf.keras.preprocessing.image.img_to_array(img)
            imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (192, 256), interpolation='bilinear')
            dataset[1].append(imgarr)
        else:
            continue
    return dataset

def getDataRandomized():
    array = []
    folders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Fake/')] 
    for folder in folders:
        images = [f for f in listdir('D:/SSD_Dataset/Images/Training/Fake/' + folder) if isfile(join('D:/SSD_Dataset/Images/Training/Fake/' + folder, f))]
        for image in images:
            array.append(['D:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + image, 1])
            
    folders = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/')]
    for folder in folders:
        images = [f for f in listdir('D:/SSD_Dataset/Images/Training/Real/' + folder) if isfile(join('D:/SSD_Dataset/Images/Training/Real/' + folder, f))]
        for image in images:
            array.append(['D:/SSD_Dataset/Images/Training/Real/' + folder + '/' + image, 0])

    array = np.array(array)
    np.random.shuffle(np.array(array))
    return array