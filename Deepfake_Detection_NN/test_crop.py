import tensorflow as tf
import numpy as np
import cv2
from utils.opencv_face_detection import cv2_face_cropper

#vairables
path_to_img = [
    "./test_data/fake_real_from_validation.jpg",
    "./test_data/real_fake_from_validation.jpg",
    "./test_data/deepfake.jpg",
    "./test_data/deepfake.png",
    "./test_data/people.jpg",
    "./test_data/fakes.jpg",
    "./test_data/morefakes.jpg",
    "./test_data/notreal.jpg",
]
models_path = [
    'M1.h5','M2.h5','M3.h5','M4.h5','M5.h5',
    'step1.h5','step2.h5','step3.h5'
]
image_resize_value = (256,256)

models = []
for path in models_path:
    models.append(tf.keras.models.load_model(path))

#get faces array
face_cropper = cv2_face_cropper()
for path in path_to_img:
    faces, img = face_cropper.getfaces_withCord(path)
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
        
        per_prediction = []
        result = 0.0
        for model in models:
            value = model.predict(test_image)[0][0]
            result += value
            per_prediction.append(str(int(result*100)))
        result /= float(len(models))

        #understanding value
        prediction = 'n/a'
        if result > 0.5:
            prediction = 'fake'
        elif result < 0.5:
            prediction = 'real'

        #output: draw rectangle, label, and log
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        color = (255,0,0)
        if 'real' in prediction:
            color = (0,255,0)
        else:
            color = (0,0,255)
        print('Prediction {}: \n{}, - {}'.format(path, prediction, result))
        cv2.putText(img, prediction+'-'+str(int(result*100))+"%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    # Display img
    cv2.imshow(path, img)

# wait until any key is pressed or close window
cv2.waitKey(0)