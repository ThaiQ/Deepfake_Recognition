from utils.predict import predict_visual,predict

#variables
#no "/" at the end
path_to_test = 'C:/Users/quach/Desktop/data_df/real_vs_fake/test/test'

path_to_img = ['./test_data/fake_real_from_validation.jpg']

model_paths = [
    './M1.h5',
    './M2.h5',
    './M3.h5',
    './M4.h5',
    './M5.h5'
]

save_path='../uploads/'

img_siz = (224,224) #224


hash,_=predict_visual(image_resize_value=img_siz, model_paths=model_paths, path_to_img=path_to_img, save=False, draw = True, show = True)
print(hash)