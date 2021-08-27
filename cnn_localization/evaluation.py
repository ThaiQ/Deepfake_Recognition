from getData import getImages_cropped, reverse_RegBox_size
import tensorflow as tf
import numpy as np
import cv2


image_resize = (100,100) #(256,256)

cnn = tf.keras.models.load_model('RCNN_97_test.h5')

test_image, image_size_ratio = getImages_cropped(['C:/Users/thai/Downloads/small_test_set/test/fake/00F8LKY6JC.jpg'], image_resize)

test_image = test_image[0]

test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)


img = cv2.imread('C:/Users/thai/Downloads/small_test_set/test/fake/00F8LKY6JC.jpg')

print(result)

regbox = reverse_RegBox_size(result[1][0],image_size_ratio[0])
x = regbox[0]
y = regbox[1]
w = regbox[2]
h = regbox[3]
print(x,y,w,h)

# cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)



# Display img
cv2.imshow('img', img)

# wait until any key is pressed or close window
cv2.waitKey(0)