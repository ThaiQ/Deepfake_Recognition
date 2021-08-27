import cv2
import matplotlib.pyplot as plt

class cv2_face_cropper():
    #loading model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    def __init__(self):
        print('Created cropper')

    def getfaces(self,img_path):
        #load image
        img = cv2.imread(img_path)
        #detect face but only in gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            image=gray, 
            scaleFactor = 1.1,
            minNeighbors = 4)

        #Processing faces and save to return array
        result_faces = []
        for (x, y, w, h) in faces:
            #crop images
            crop = img[y:y+h, x:x+w]
            result_faces.append(crop)

        return result_faces
    
    def getfaces_img(self,img):
        #detect face but only in gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            image=gray, 
            scaleFactor = 1.1,
            minNeighbors = 4)

        #Processing faces and save to return array
        result_faces = []
        for (x, y, w, h) in faces:
            #crop images
            crop = img[y:y+h, x:x+w]
            result_faces.append(crop)

        return result_faces
    
    def getfaces_withCord(self,img_path):
        #load image
        img = cv2.imread(img_path)
        #detect face but only in gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            image=gray, 
            scaleFactor = 1.1,
            minNeighbors = 4)

        #Processing faces and save to return array
        result_faces = []
        for (x, y, w, h) in faces:
            #crop images
            crop = img[y:y+h, x:x+w]
            result_faces.append({
                'img': crop,
                'x' : x,
                'y' : y,
                'w' : w,
                'h' : h,
            })
        return result_faces, img
    
    def display_faces(self, faces):
        ind=0
        for face in faces:
            cv2.imshow('face'+str(ind), face)
            ind+=1
        cv2.waitKey(0)

'''
image_set returned from tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory()
'''
def display_keras_imageGenerator(image_set):
    for _ in range(image_set.batch_size):
        img, label = image_set.next()
    print(img.shape, label)   #  (1,256,256,3)
    plt.imshow(img[0])
    plt.show()
