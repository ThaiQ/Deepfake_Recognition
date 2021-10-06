from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np
import cv2
from utils.opencv_face_detection import cv2_face_cropper

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
    face_cropper = cv2_face_cropper()
    dataset = [[], []]
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder)]
        if len(images) > 0:
            faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + images[0])
            if len(faces[0]) == 1:
                test_image = cv2.resize(faces[0][0]['img'], (224, 224))
                test_image = (test_image)/255.0
                dataset[1].append(test_image)
        else:
            continue
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')] #skip over .9 of the original images from fake, the other .1 from real
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + folder)]
        if len(images) > 0:
            faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/Training/real/' + folder + '/' + images[0])
            if len(faces[0]) == 1:
                test_image = cv2.resize(faces[0][0]['img'], (224, 224))
                test_image = (test_image)/255.0
                dataset[0].append(test_image)
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
    np.random.shuffle(array)
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

def getValidationData_path(dir_validation='C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/valid', resize_target=(64, 64)):
    batch = [[], []]
    images = [f for f in listdir(dir_validation+'/fake')]
    for image in images:
        batch[0].append(dir_validation+'/fake/' + image)
    images = [f for f in listdir(dir_validation+'/real')]
    for image in images:
        batch[1].append(dir_validation+'/real/' + image)
    return batch

def getValidationData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Combined_Dataset/Validation/V1/Real/')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Combined_Dataset/Validation/V1/Real/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = imgarr/255.0
        batch[0].append(imgarr)
            
    images = [f for f in listdir('C:/SSD_Dataset/Combined_Dataset/Validation/V1/Fake/')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Combined_Dataset/Validation/V1/Fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = imgarr/255.0
        batch[1].append(imgarr)
    return batch


def getV2ValidationData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Combined_Dataset/Validation/V2/Fake')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Combined_Dataset/Validation/V2/Fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = imgarr/255.0
        batch[1].append(imgarr)
    images = [f for f in listdir('C:/SSD_Dataset/Combined_Dataset/Validation/V2/Real')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Combined_Dataset/Validation/V2/Real/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = imgarr/255.0
        batch[0].append(imgarr)
    return batch

def getV2ValidationDataCropped(dir='C:/SSD_Dataset/Images/V2/valid',img_size=(224,224)):
    face_cropper = cv2_face_cropper()
    dataset = [[], []]
    images = [f for f in listdir(dir+'/fake')]
    for image in images:
        faces = face_cropper.getfaces_withCord(dir+'/fake/' + image)
        if len(faces[0]) == 1:
            test_image = cv2.resize(faces[0][0]['img'], img_size)
            test_image = (test_image)/255.0
            dataset[1].append(test_image)
    images = [f for f in listdir(dir+'/real')]
    for image in images:
        faces = face_cropper.getfaces_withCord(dir+'/real/' + image)
        if len(faces[0]) == 1:
            test_image = cv2.resize(faces[0][0]['img'], img_size)
            test_image = (test_image)/255.0
            dataset[0].append(test_image)
    return dataset

def getV3ValidationData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Images/V3/real_and_fake_face/training_fake')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/V3/real_and_fake_face/training_fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (224, 224), interpolation='bilinear')
        batch[1].append(imgarr)
    images = [f for f in listdir('C:/SSD_Dataset/Images/V3/real_and_fake_face/training_real')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/V3/real_and_fake_face/training_real/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        imgarr = tf.keras.preprocessing.image.smart_resize(imgarr, (224, 224), interpolation='bilinear')
        batch[0].append(imgarr)
    return batch

def getV2TestData():
    batch = [[], []]
    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/test/fake')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/V2/test/fake/' + image)
        imgarr = tf.keras.preprocessing.image.img_to_array(img)
        batch[1].append(imgarr)
    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/test/real')]
    for image in images:
        img = tf.keras.preprocessing.image.load_img('C:/SSD_Dataset/Images/V2/test/real/' + image)
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

def getCombinedDatasetRandomized():
    array = []
    images = [f for f in listdir('C:/SSD_Dataset/Combined_Dataset/Training/Fake')]
    for image in images:
        array.append(['C:/SSD_Dataset/Combined_Dataset/Training/Fake/' + image, 1])
            
    images = [f for f in listdir('C:/SSD_Dataset/Combined_Dataset/Training/Real')]
    for image in images:
        array.append(['C:/SSD_Dataset/Combined_Dataset/Training/Real/' + image, 0])

    array = np.array(array)
    np.random.shuffle(array)
    return array

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

def getDataFromListCropped(filelist):
    face_cropper = cv2_face_cropper()
    dataset = [[], []]
    for file in filelist:
        faces = face_cropper.getfaces_withCord(file[0])
        if len(faces[0]) == 1:
            test_image = cv2.resize(faces[0][0]['img'], (256, 256))
            #cv2.imshow('img', test_image)
            test_image = (test_image)/255.0
            dataset[0].append(test_image)
            dataset[1].append(file[1].astype(np.float))
    return dataset

def cropImages():
    face_cropper = cv2_face_cropper()
    realcount = 0 #48436
    fakecount = 0 #55291

    # images = [f for f in listdir('C:/SSD_Dataset/Images/V2/train/fake')]
    # for image in images:
    #     faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/V2/train/fake/' + image)
    #     if len(faces[0]) == 1:
    #         cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
    #         cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Training/Fake/fake_" + str(fakecount) + ".jpg", cropped_image)
    #         fakecount += 1

    # images = [f for f in listdir('C:/SSD_Dataset/Images/V2/train/real')]
    # for image in images:
    #     faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/V2/train/real/' + image)
    #     if len(faces[0]) == 1:
    #         cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
    #         cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Training/Real/real_" + str(realcount) + ".jpg", cropped_image)
    #         realcount += 1

    # folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')] #skip over .9 of the original images from fake, the other .1 from real
    # for folder in folders:
    #     images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder)]
    #     if len(images) > 0:
    #         faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + images[0])
    #         if len(faces[0]) == 1:
    #             cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
    #             cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Training/Fake/fake_" + str(fakecount) + ".jpg", cropped_image)
    #             fakecount += 1
    #     else:
    #         continue
    # folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')] #skip over .9 of the original images from fake, the other .1 from real
    # for folder in folders:
    #     images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + folder)]
    #     if len(images) > 0:
    #         faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/Training/real/' + folder + '/' + images[0])
    #         if len(faces[0]) == 1:
    #             cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
    #             cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Training/Real/real_" + str(realcount) + ".jpg", cropped_image)
    #             realcount += 1
    #     else:
    #         continue

    # images = [f for f in listdir('C:/SSD_Dataset/Images/V3/real_and_fake_face_detection/real_and_fake_face/training_fake')]
    # for image in images:
    #     faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/V3/real_and_fake_face_detection/real_and_fake_face/training_fake/' + image)
    #     if len(faces[0]) == 1:
    #         cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
    #         cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Training/Fake/fake_" + str(fakecount) + ".jpg", cropped_image)
    #         fakecount += 1

    # images = [f for f in listdir('C:/SSD_Dataset/Images/V3/real_and_fake_face_detection/real_and_fake_face/training_real')]
    # for image in images:
    #     faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/V3/real_and_fake_face_detection/real_and_fake_face/training_real/' + image)
    #     if len(faces[0]) == 1:
    #         cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
    #         cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Training/Real/real_" + str(realcount) + ".jpg", cropped_image)
    #         realcount += 1

    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/valid/fake')]
    for image in images:
        faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/V2/valid/fake/' + image)
        if len(faces[0]) == 1:
            cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
            cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Validation/V2/Fake/fake_" + str(fakecount) + ".jpg", cropped_image)
            fakecount += 1

    images = [f for f in listdir('C:/SSD_Dataset/Images/V2/valid/real')]
    for image in images:
        faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/V2/valid/real/' + image)
        if len(faces[0]) == 1:
            cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
            cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Validation/V2/Real/real_" + str(realcount) + ".jpg", cropped_image)
            realcount += 1
    
    fakecount = 0
    realcount = 0
    
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/')] 
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Fake/' + folder)]
        if len(images) > 0:
            faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/Training/Fake/' + folder + '/' + images[0])
            if len(faces[0]) == 1:
                cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
                cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Validation/V1/Fake/fake_" + str(fakecount) + ".jpg", cropped_image)
                fakecount += 1
        else:
            continue
    folders = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/')] 
    for folder in folders:
        images = [f for f in listdir('C:/SSD_Dataset/Images/Training/Real/' + folder)]
        if len(images) > 0:
            faces = face_cropper.getfaces_withCord('C:/SSD_Dataset/Images/Training/real/' + folder + '/' + images[0])
            if len(faces[0]) == 1:
                cropped_image = cv2.resize(faces[0][0]['img'], (256, 256))
                cv2.imwrite("C:/SSD_Dataset/Combined_Dataset/Validation/V1/Real/real_" + str(realcount) + ".jpg", cropped_image)
                realcount += 1
        else:
            continue

    print(realcount)
    print(fakecount)

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

