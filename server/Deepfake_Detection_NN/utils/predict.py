import tensorflow as tf
import numpy as np
import cv2
from Deepfake_Detection_NN.utils.opencv_face_detection import cv2_face_cropper
import hashlib
from Deepfake_Detection_NN.getData import getV2ValidationDataCropped
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn

def predict_visual(image_resize_value=(224,224), model_paths=[], path_to_img=[], 
save=None, draw = True, show = True):
    if not isinstance(model_paths, list) and not isinstance(path_to_img, list):
        print("model_paths or path_to_imgs are arrays ['path']")
        return None
    elif isinstance(model_paths, list) and len(model_paths)<1:
        print("model_paths is empty")
        return
    elif isinstance(path_to_img, list) and len(path_to_img)<1:
        print("path_to_img is empty")
        return

    models = []
    for path in model_paths:
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
            
            if draw:
                #output: draw rectangle, label, and log
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                color = (255,0,0)
                if 'real' in prediction:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                print('Prediction {}: \n{}, - {}'.format(path, prediction, result))
                cv2.putText(img, prediction+'-'+str(int(result*100))+"%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if save is not None or save is not False:
            save_dir = ""
            hash = hashlib.sha224(path.encode()).hexdigest()
            if save is None or save is True: save_dir = "./"+hash+".png"
            else: save_dir = save+hash+".png"
            print(save_dir)
            cv2.imwrite(save_dir,img)

        if show:
            # Display img
            cv2.imshow(path, img)

    if show:
        # wait until any key is pressed or close window
        cv2.waitKey(0)
    
    return hash, img

def predict(image_resize_value=(224,224), model_paths=[], path_to_test_set='C:/Users/quach/Desktop/data_df/real_vs_fake/test/test', show = False):
    if not isinstance(model_paths, list):
        print("model_paths or path_to_imgs are arrays ['path']")
        return None
    elif isinstance(model_paths, list) and len(model_paths)<1:
        print("model_paths is empty")
        return

    models = []
    for path in model_paths:
        models.append(tf.keras.models.load_model(path))

    batches = getV2ValidationDataCropped(path_to_test_set,image_resize_value)

    X_fake = np.array(batches[1])
    X_real = np.array(batches[0])

    fake_preds = np.array([0]*len(X_fake))
    real_preds = np.array([0]*len(X_real))

    fake_records = []
    real_records = []

    for model in models:
        #prediction
        pf = np.array(model.predict(X_fake, batch_size=32)).flatten()
        fake_preds = np.add(fake_preds,pf)
        pr = np.array(model.predict(X_real, batch_size=32)).flatten()
        real_preds = np.add(real_preds,pr)
        #record
        fake_records.append(pf)
        real_records.append(pr)


    
    fake_preds=fake_preds/len(models)
    real_preds=real_preds/len(models)

    expected_fake_labels = []
    expected_real_labels = []

    for fake, real in zip(fake_preds,real_preds):
        if fake > 0.5: expected_fake_labels.append(1)
        else: expected_fake_labels.append(0)
        if real > 0.5: expected_real_labels.append(1)
        else: expected_real_labels.append(0)

    #plot confusion
    conf_mtx = confusion_matrix(
        ([1]*len(expected_fake_labels))+([0]*len(expected_real_labels)),
        expected_fake_labels+expected_real_labels,)
    conf_mtx_norm = confusion_matrix(
        ([1]*len(expected_fake_labels))+([0]*len(expected_real_labels)),
        expected_fake_labels+expected_real_labels,
        normalize='true')
    plot_confusion_matrix(conf_mtx, ["real","fake"])
    plot_confusion_matrix(conf_mtx_norm, ["real","fake"],"Normalized Confusion Matrix")
    #ROC curve
    plot_ROC_curve(fake_preds, real_preds)
    plot_ROC_curve_overlay(fake_records, real_records)

    if show: plt.show()

    return fake_preds,real_preds,expected_fake_labels,expected_real_labels


def plot_confusion_matrix(data, labels, title = "Confusion Matrix"):
        """Plot confusion matrix using heatmap.
        Args:
            data (list of list): List of lists with confusion matrix data.
            labels (list): Labels which will be plotted across x and y axis.
            output_filename (str): Path to output file.
        """
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1)

        seaborn.set(color_codes=True)
        seaborn.set(font_scale=1.4)
        ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, ax=ax)

        ax.set_xticks([0.5,1.5])
        ax.set_yticks([0.5,1.5])

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
        ax.set(ylabel="True Label", xlabel="Predicted Label", title = title)
    
def plot_ROC_curve_overlay(fake_records, real_records):
        """Plot ROC curve.
        Args:
            fake_records (2d list): prediction of fake of each model.
            real_records (2d list): prediction of real of each model.
        """
        
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,1],[0,1],linestyle='--', label='0.5 line')

        model_ind = 1
        for fake_preds,real_preds in zip(fake_records, real_records):
            fpr, tpr, _ = roc_curve(
                ([1]*len(fake_preds))+([0]*len(real_preds)), 
                list(fake_preds)+list(real_preds), pos_label=0)
            #auc score
            auc=roc_auc_score(([1]*len(fake_preds))+([0]*len(real_preds)), list(fake_preds)+list(real_preds))
            ax.plot(tpr,fpr, marker='.', label='Model-{} AUC: {}'.format(model_ind, auc))
            model_ind+=1

        ax.set(ylabel="True positive rate", xlabel="False positive rate", title="ROC curve of models")
        ax.legend()

def plot_ROC_curve(fake_preds, real_preds):
        """Plot ROC curve.
        Args:
            fake_preds (list): prediction of fake.
            real_preds (list): prediction of real.
        """
        fpr, tpr, _ = roc_curve(
            ([1]*len(fake_preds))+([0]*len(real_preds)), 
            list(fake_preds)+list(real_preds), pos_label=0)
        #auc score
        auc=roc_auc_score(([1]*len(fake_preds))+([0]*len(real_preds)), list(fake_preds)+list(real_preds))
        
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,1],[0,1],linestyle='--', label='0.5 line')
        ax.plot(tpr,fpr, marker='.', label='Model AUC: {}'.format(auc))
        ax.set(ylabel="True positive rate", xlabel="False positive rate", title="Ensemble ROC curve")
        ax.legend()
