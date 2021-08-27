from thispersondoesnotexist import get_online_person, get_checksum_from_picture, save_picture
import os
import csv
import cv2
from utils.opencv_face_detection import cv2_face_cropper

#variable
image_counts=5 #saving 20k images

#newfolder
newpath = r'./DoesnotExistData' 
csv_path = "./DoesnotExistData/DoesnotExistData_filename.csv"
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(newpath+'./fake')
    os.makedirs(newpath+'./real')

saved = []
f = open(csv_path, "a")
with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            saved.append(row[0])

i = 0
cropper = cv2_face_cropper()
while (i < image_counts):
    # Using function
    picture = get_online_person()
    checksum2 = get_checksum_from_picture(picture)  # Method is optional, defaults to "md5"

    if checksum2 not in saved:
        saved.append(checksum2)
        f.write(checksum2+'\n')

        save_picture(picture, './DoesnotExistData/fake/{}.jpg'.format(checksum2))
        picture = cropper.getfaces('./DoesnotExistData/fake/{}.jpg'.format(checksum2))[0]
        cv2.imwrite('./DoesnotExistData/fake/{}.jpg'.format(checksum2), picture)

        i += 1
    
    print('{} / {}'.format(i,image_counts))

f.close()