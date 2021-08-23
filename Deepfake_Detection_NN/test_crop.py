import tensorflow as tf
import numpy as np
import cv2
from utils.opencv_face_detection import cv2_face_cropper

#vairables
path_to_img = './test_data/fake_real_from_validation.jpg'
image_resize_value = (256,256)
model = '98_99_V2_Deepfake_Detector_Model_david.h5'

#get faces array
face_cropper = cv2_face_cropper()
faces, img = face_cropper.getfaces_withCord(path_to_img)

# Recreate the exact same model, including its weights and the optimizer
cnn = tf.keras.models.load_model(model)
# Show the model architecture
cnn.summary()

for face in faces:
    x = face['x']
    y = face['y']
    w = face['w']
    h = face['h']

    #processing and prediction
    test_image = cv2.resize(face['img'], image_resize_value)
    test_image = tf.convert_to_tensor(test_image, dtype=tf.float32)
    test_image = (test_image)/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)

    #understanding value
    prediction = 'n/a'
    if 'david' in model and result[0][0] > 0.5:
        prediction = 'fake'
    elif 'david' in model and result[0][0] < 0.5:
        prediction = 'real'
    elif 'thai' in model and result[0][0] < 0.5:
        prediction = 'fake'
    elif 'thai' in model and result[0][0] > 0.5:
        prediction = 'real'

    #output: draw rectangle, label, and log
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if 'real' in prediction:
        cv2.putText(img, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
        cv2.putText(img, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    print('Prediction: {}, - {}'.format(prediction, result))


# Display img
cv2.imshow('img', img)
# wait until any key is pressed or close window
cv2.waitKey(0)