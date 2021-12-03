from cv2 import cv2
img = cv2.imread("C:\\Github\\Deepfake_Recognition_SSD\\SSD_Implement_v2\\test_data\\JPEGImages\\id0_id1_0000_2.jpg")
img[108:214, 501] = (255,0,255)
cv2.imwrite('temp.jpg', img)
print('done')