from Deepfake_Detection_NN.utils.predict import predict
from os import listdir

#variables (can change)
#no "/" at the end
path_to_test = 'C:/Users/thai/Desktop/ssd/david/Deepfake_Detection_NN/test_data'
img_siz = (224,224) #224


#load models from 'models' folder (don't change)
models = ['./models/'+f for f in listdir('./models')]
models = list(filter(lambda f: f[-2:] == 'h5', models))
#execute
fake_preds,real_preds,expected_fake_labels,expected_real_labels=predict(image_resize_value=(224,224), model_paths=models, path_to_test_set=path_to_test, show = True)
