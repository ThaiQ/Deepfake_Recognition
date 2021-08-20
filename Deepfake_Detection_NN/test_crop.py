from utils.opencv_face_detection import cv2_face_cropper

#get faces array
face_cropper = cv2_face_cropper()
faces = face_cropper.getfaces('./test_data/people.jpg')

#display
face_cropper.display_faces(faces)