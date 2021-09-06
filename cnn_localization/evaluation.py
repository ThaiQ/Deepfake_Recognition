from getData import getImages_data
import tensorflow as tf
import numpy as np
import cv2

image_resize = (200,200) #(256,256)

cnn = tf.keras.models.load_model('RCNN.h5')

filelist = [
    'C:/Users/thai/Downloads/small_test_set/test/fake/00V5CZZSSO.jpg',
    'C:/Users/thai/Downloads/small_test_set/test/fake/00XUQJZGHU.jpg',
    'C:/Users/thai/Downloads/small_test_set/test/real/00001.jpg',
    'C:/Users/thai/Downloads/small_test_set/test/real/00004.jpg',
    '../Deepfake_Detection_NN/test_data/deepfake.png',
]

#preprocessing
image_data = getImages_data(filelist, img_size=image_resize)
image_data = np.expand_dims(image_data, axis = 0)
results = cnn.predict(np.vstack(image_data))

#output cv2
fake_val = results[0]
regCord = results[1]
for val, cord, path in zip(fake_val, regCord, filelist):
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    x = int(cord[0]*w)
    y = int(cord[1]*h)
    w = int(cord[2]*w)
    h = int(cord[3]*h)
    
    #understanding value
    prediction = 'n/a'
    if val > 0.5:
        prediction = 'fake'
    elif val < 0.5:
        prediction = 'real'

    #writing to img
    color = (255,0,0)
    if 'real' in prediction:
        color = (0,255,0)
    else:
        color = (0,0,255)
    print('{} - {}'.format(path, val))
    
    cv2.putText(img, prediction+'-'+str(int(val*100))+"%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('img '+str(path), img)

# wait until any key is pressed or close window
cv2.waitKey(0)