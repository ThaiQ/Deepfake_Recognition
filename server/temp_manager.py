import datetime
from Deepfake_Detection_NN.utils.predict import predict_visual
from os import listdir

class Temp_Manager:
    memo = {
        "filename": { #sample format
            "predicted_file": "./uploads/hash.png",
            "ts": datetime.datetime.utcnow()
        }
    }

    models = ['./models/'+f for f in listdir('./models')]
    models = list(filter(lambda f: f[-2:] == 'h5', models))

    save_path='./uploads/'

    img_siz = (224,224)

    def process(self, img_name):

        hashFile = ''
        if img_name in self.memo:
            pointer = self.memo[img_name]
            hashFile = pointer["predicted_file"]
        else:
            img_path = './uploads/'+img_name
            extension = img_name[img_name.index('.'):]
            hash,_=predict_visual(image_resize_value=self.img_siz, model_paths=self.models, path_to_img=[img_path], save=self.save_path, draw = True, show = False)
            if extension == '.mp4':
                hashFile = hash+extension
            else:
                hashFile = hash+".png"
            self.memo[img_name] = {
                "predicted_file": hashFile,
                "ts": datetime.datetime.utcnow()
            }

        #remove old memory
        print(self.memo)
        del_key = []
        for key in self.memo:
            item = self.memo[key]
            ts = item['ts']
            ts = datetime.datetime.utcnow() - ts
            hr = str(ts).split(':')[0]
            if int(hr) >= 120: #delete 5 days old keys
                del_key.append(key)
        for key in del_key: del self.memo[key]

        return img_name,hashFile
    

