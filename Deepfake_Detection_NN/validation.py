import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from getData import getValidationData_path
import numpy as np

batch_size = 10000 #10,000
image_resize = (256,256) #(256,256)

test_model_with_dataset = [
    ('93_Deepfake_Detector_Model_from_train2_thai.h5', 'valid'),
    ('95_Deepfake_Detector_Model_from_train2_thai.h5', 'valid'),
    ('98_99_V2_Deepfake_Detector_Model_david.h5', 'test'),

    ('93_Deepfake_Detector_Model_from_train2_thai.h5', 'test'),
    ('95_Deepfake_Detector_Model_from_train2_thai.h5', 'test'),
    ('98_99_V2_Deepfake_Detector_Model_david.h5', 'valid')
]

def test(model, folder):
    # Recreate the exact same model, including its weights and the optimizer
    cnn = tf.keras.models.load_model(model)
    # Show the model architecture
    cnn.summary()

    batches = getValidationData_path('C:/Users/quach/Desktop/data_df/real_vs_fake/real-vs-fake/'+folder)
    length = len(batches[0])+len(batches[1])
    current=0

    positive_fake = 0
    for fake in batches[0]:
        current+=1
        test_image = tf.keras.preprocessing.image.load_img(fake, target_size = (256, 256))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = (test_image.astype(np.float))/255.0
        #test_image = tf.keras.applications.resnet50.preprocess_input(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        prediction = 'n/a'
        if 'david' in model and result[0][0] > 0.5:
            prediction = 'fake'
            positive_fake += 1
        elif 'david' in model and result[0][0] < 0.5:
            prediction = 'real'
        elif 'thai' in model and result[0][0] < 0.5:
            prediction = 'fake'
            positive_fake += 1
        elif 'thai' in model and result[0][0] > 0.5:
            prediction = 'real'
        print('Prediction: {}, expected: {}, status: {}/{}'.format(prediction,'fake',current,length))

    positive_real = 0
    for real in batches[1]:
        current+=1
        test_image = tf.keras.preprocessing.image.load_img(real, target_size = (256, 256))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = (test_image.astype(np.float))/255.0
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        prediction = 'n/a'
        if 'david' in model and result[0][0] > 0.5:
            prediction = 'fake'
        elif 'david' in model and result[0][0] < 0.5:
            prediction = 'real'
            positive_real += 1
        if 'thai' in model and result[0][0] < 0.5:
            prediction = 'fake'
        elif 'thai' in model and result[0][0] > 0.5:
            prediction = 'real'
            positive_real += 1
        print('Prediction: {}, expected: {}, status: {}/{}'.format(prediction,'real',current,length))

    total_fake=len(batches[0])
    total_real=len(batches[1])

    report = '''
===================================================================
{title}
===================================================================
Result:
positive_fake / total_fake: {positive_fake} / {total_fake}
accuracy_fake: {accuracy_fake}%

positive_real / total_real: {positive_real} / {total_real}
accuracy_real: {accuracy_real}%
====================================================================

    '''.format(
        title = model + ' with ' + folder,

        positive_fake=positive_fake,
        accuracy_fake=((positive_fake/total_fake)*100),
        total_fake=total_fake,

        positive_real=positive_real,
        accuracy_real=((positive_real/total_real)*100),
        total_real=total_real
    )

    print(report)

    f = open("test_result.md", "a")
    f.write(report)
    f.close()


#Execute
for trial in test_model_with_dataset:
    test(trial[0], trial[1])