from os import listdir
from utils.opencv_face_detection import cv2_face_cropper

def main(dir = 'C:/Users/thai/Downloads/small_test_set/test'):
    f = open("converted_path.csv", "a")
    cropper = cv2_face_cropper()
    count = 0

    images_fake = [f for f in listdir(dir+'/fake')]
    images_real = [f for f in listdir(dir+'/real')]
    length=len(images_fake)+len(images_real)

    for image in images_fake:
        count+=1
        img_path = dir+'/fake/' + image
        faces,_=cropper.getfaces_withCord(img_path)
        for face in faces:
            data='{},{},{},{},{},{}\n'.format(
                img_path,
                1,
                face['x'],
                face['y'],
                face['w'],
                face['h']
                )
            f.write(data)
            print('{} / {}'.format(count,length))

    for image in images_real:
        count+=1
        img_path = dir+'/real/' + image
        faces,_=cropper.getfaces_withCord(img_path)
        for face in faces:
            data='{},{},{},{},{},{}\n'.format(
                img_path,
                0,
                face['x'],
                face['y'],
                face['w'],
                face['h']
                )
            f.write(data)
            print('{} / {}'.format(count,length))
        
    f.close()
    print('Converted {} files'.format(count))
    return

#execute
main(dir = 'C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/train')