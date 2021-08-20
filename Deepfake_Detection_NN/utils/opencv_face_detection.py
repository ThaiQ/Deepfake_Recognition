import cv2

# #loading model
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# #load image
# # img = cv2.imread('./test_data/id0_0001/id0_0001_0.jpg')
# img = cv2.imread('./test_data/people.jpg')
# #detect face but only in gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(
#     image=gray, 
#     scaleFactor = 1.1,
#     minNeighbors = 4)

# #Processing faces and save to return array
# result_faces = []
# for (x, y, w, h) in faces:
#     #crop images
#     crop = img[y:y+h, x:x+w]
#     result_faces.append(crop)

# cv2.imshow('imgcolor', img)

# cv2.imshow('img', gray)

# ind=0
# for face in result_faces:
#     cv2.imshow('img'+str(ind), face)
#     ind+=1

# cv2.waitKey(0)

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
    
    def display_faces(self, faces):
        ind=0
        for face in faces:
            cv2.imshow('face'+str(ind), face)
            ind+=1
        cv2.waitKey(0)
