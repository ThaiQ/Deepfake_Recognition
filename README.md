# SJSU_Cmpe195F - Fall 2021 - Deepfake Detection Project

# Dependencies
1. NodeJS: https://nodejs.org/en/download/
2. Python3: https://www.python.org/downloads/
3. Tensorflow 2.0 or Cuda if GPU (Optional)

# Description
This is a student project that aims to build a deepfake detection web application.

High level Architecture: React-Flask stack

![Project Deployment](https://user-images.githubusercontent.com/18486562/143810257-6e65f424-4852-4b2b-a591-cbadf70fc096.png)

Design Diagram

![design (4)](https://user-images.githubusercontent.com/18486562/143810327-9ff49df8-fd94-427c-8291-97c3f11c3820.png)

CNN Architecture

![unnamed](https://user-images.githubusercontent.com/18486562/143810556-560ec505-c388-43c5-a564-87e55adb9568.png)

# Server installation
1. CD to server
2. `pip install -r ./Deepfake_Detection_NN/requirement.txt`
3. Download our ensemble models
4. Save [downloaded](https://drive.google.com/file/d/1wjqRtgTM5mWk4wyAUApbsdG1HMYjgWmt/view?usp=sharing) models in `server/models` - Make sure its `.h5` files - script will handle parsing.
   
![image](https://user-images.githubusercontent.com/18486562/137400179-1abc726e-1d04-407f-b194-2f9c1a6ea66b.png)

5. Run server `python app.py`

# Frontend installation
1. CD to app
2. `npm install` - install node dependencies
3. Run react app `npm start`

# If you want to train your own model:
1. Follow `server installation` to step 2
3. Get data at https://www.kaggle.com/c/deepfake-detection-challenge/data
2. run `train.py` in `./server/Deepfake_Detection_NN`
