from utils.predict import predict_visual,predict


#variables

#no "/" at the end
path_to_test = 'C:/Users/quach/Desktop/data_df/real_vs_fake/test/test'

model_paths = [
    './M1.h5',
    './M2.h5'
    './M3.h5'
    './M4.h5'
    './M5.h5'
]

img_siz = (256,256)


predict(image_resize_value=img_siz, model_paths=model_paths, path_to_test_set=path_to_test, show = True)