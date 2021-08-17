Model: "DeepfakeDetector"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 256, 256, 3)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 125, 125, 16)      2368
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 125, 125, 32)      4640
_________________________________________________________________
batch_normalization (BatchNo (None, 125, 125, 32)      128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 62, 62, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 62, 62, 32)        9248
_________________________________________________________________
batch_normalization_1 (Batch (None, 62, 62, 32)        128
_________________________________________________________________
dropout (Dropout)            (None, 62, 62, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 62, 62, 64)        18496
_________________________________________________________________
batch_normalization_2 (Batch (None, 62, 62, 64)        256
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 31, 31, 64)        36928
_________________________________________________________________
batch_normalization_3 (Batch (None, 31, 31, 64)        256
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 31, 64)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 16, 128)       73856
_________________________________________________________________
batch_normalization_4 (Batch (None, 16, 16, 128)       512
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 128)         147584
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 8, 128)         512
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0
_________________________________________________________________
dense (Dense)                (None, 1024)              8389632
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 1025
=================================================================
Total params: 8,685,569
Trainable params: 8,684,673
Non-trainable params: 896
_________________________________________________________________
0.4917
2021-08-16 20:51:33.696663: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/10
2021-08-16 20:51:34.247768: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-08-16 20:51:35.160221: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-08-16 20:51:36.108734: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-08-16 20:51:38.646567: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2021-08-16 20:51:38.719621: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2021-08-16 20:51:42.195692: I tensorflow/stream_executor/cuda/cuda_blas.cc:1838] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
10/10 [==============================] - 15s 436ms/step - loss: 0.9973 - binary_accuracy: 0.5277
Epoch 2/10
10/10 [==============================] - 4s 441ms/step - loss: 0.4583 - binary_accuracy: 0.6212
Epoch 3/10
10/10 [==============================] - 4s 420ms/step - loss: 0.3105 - binary_accuracy: 0.6750
Epoch 4/10
10/10 [==============================] - 4s 411ms/step - loss: 0.2823 - binary_accuracy: 0.7197
Epoch 5/10
10/10 [==============================] - 4s 409ms/step - loss: 0.2543 - binary_accuracy: 0.7518
Epoch 6/10
10/10 [==============================] - 4s 412ms/step - loss: 0.2294 - binary_accuracy: 0.7825
Epoch 7/10
10/10 [==============================] - 4s 406ms/step - loss: 0.2054 - binary_accuracy: 0.8095
Epoch 8/10
10/10 [==============================] - 4s 427ms/step - loss: 0.1858 - binary_accuracy: 0.8324
Epoch 9/10
10/10 [==============================] - 4s 437ms/step - loss: 0.1689 - binary_accuracy: 0.8480
Epoch 10/10
10/10 [==============================] - 4s 369ms/step - loss: 0.1471 - binary_accuracy: 0.8739
0.5023
Epoch 1/10
10/10 [==============================] - 5s 474ms/step - loss: 0.2344 - binary_accuracy: 0.7844
Epoch 2/10
10/10 [==============================] - 5s 508ms/step - loss: 0.1946 - binary_accuracy: 0.8276
Epoch 3/10
10/10 [==============================] - 5s 466ms/step - loss: 0.1607 - binary_accuracy: 0.8588
Epoch 4/10
10/10 [==============================] - 4s 439ms/step - loss: 0.1337 - binary_accuracy: 0.8836
Epoch 5/10
10/10 [==============================] - 4s 442ms/step - loss: 0.1118 - binary_accuracy: 0.9087
Epoch 6/10
10/10 [==============================] - 4s 432ms/step - loss: 0.0866 - binary_accuracy: 0.9302
Epoch 7/10
10/10 [==============================] - 4s 410ms/step - loss: 0.0738 - binary_accuracy: 0.9443
Epoch 8/10
10/10 [==============================] - 4s 437ms/step - loss: 0.0626 - binary_accuracy: 0.9520
Epoch 9/10
10/10 [==============================] - 4s 372ms/step - loss: 0.0504 - binary_accuracy: 0.9623
Epoch 10/10
10/10 [==============================] - 3s 345ms/step - loss: 0.0436 - binary_accuracy: 0.9705
0.5012
Epoch 1/10
10/10 [==============================] - 5s 429ms/step - loss: 0.1998 - binary_accuracy: 0.8394
Epoch 2/10
10/10 [==============================] - 4s 449ms/step - loss: 0.1403 - binary_accuracy: 0.8872
Epoch 3/10
10/10 [==============================] - 4s 410ms/step - loss: 0.1054 - binary_accuracy: 0.9174
Epoch 4/10
10/10 [==============================] - 4s 403ms/step - loss: 0.0731 - binary_accuracy: 0.9437
Epoch 5/10
10/10 [==============================] - 4s 401ms/step - loss: 0.0556 - binary_accuracy: 0.9604
Epoch 6/10
10/10 [==============================] - 4s 401ms/step - loss: 0.0436 - binary_accuracy: 0.9688
Epoch 7/10
10/10 [==============================] - 4s 417ms/step - loss: 0.0338 - binary_accuracy: 0.9739
Epoch 8/10
10/10 [==============================] - 4s 391ms/step - loss: 0.0274 - binary_accuracy: 0.9816
Epoch 9/10
10/10 [==============================] - 4s 363ms/step - loss: 0.0246 - binary_accuracy: 0.9850
Epoch 10/10
10/10 [==============================] - 4s 369ms/step - loss: 0.0184 - binary_accuracy: 0.9879
0.4963
Epoch 1/10
10/10 [==============================] - 5s 422ms/step - loss: 0.1684 - binary_accuracy: 0.8754
Epoch 2/10
10/10 [==============================] - 4s 438ms/step - loss: 0.1049 - binary_accuracy: 0.9155
Epoch 3/10
10/10 [==============================] - 4s 429ms/step - loss: 0.0698 - binary_accuracy: 0.9446
Epoch 4/10
10/10 [==============================] - 4s 392ms/step - loss: 0.0449 - binary_accuracy: 0.9667
Epoch 5/10
10/10 [==============================] - 4s 408ms/step - loss: 0.0346 - binary_accuracy: 0.9735
Epoch 6/10
10/10 [==============================] - 4s 400ms/step - loss: 0.0249 - binary_accuracy: 0.9841
Epoch 7/10
10/10 [==============================] - 4s 409ms/step - loss: 0.0189 - binary_accuracy: 0.9891
Epoch 8/10
10/10 [==============================] - 4s 419ms/step - loss: 0.0154 - binary_accuracy: 0.9902
Epoch 9/10
10/10 [==============================] - 4s 391ms/step - loss: 0.0130 - binary_accuracy: 0.9919
Epoch 10/10
10/10 [==============================] - 4s 374ms/step - loss: 0.0101 - binary_accuracy: 0.9951
0.5064
Epoch 1/10
10/10 [==============================] - 5s 441ms/step - loss: 0.1530 - binary_accuracy: 0.8904
Epoch 2/10
10/10 [==============================] - 4s 427ms/step - loss: 0.0894 - binary_accuracy: 0.9298
Epoch 3/10
10/10 [==============================] - 4s 394ms/step - loss: 0.0589 - binary_accuracy: 0.9556
Epoch 4/10
10/10 [==============================] - 4s 415ms/step - loss: 0.0320 - binary_accuracy: 0.9785
Epoch 5/10
10/10 [==============================] - 4s 432ms/step - loss: 0.0244 - binary_accuracy: 0.9824
Epoch 6/10
10/10 [==============================] - 4s 397ms/step - loss: 0.0171 - binary_accuracy: 0.9898
Epoch 7/10
10/10 [==============================] - 4s 409ms/step - loss: 0.0166 - binary_accuracy: 0.9894
Epoch 8/10
10/10 [==============================] - 4s 406ms/step - loss: 0.0130 - binary_accuracy: 0.9916
Epoch 9/10
10/10 [==============================] - 4s 373ms/step - loss: 0.0129 - binary_accuracy: 0.9907
Epoch 10/10
10/10 [==============================] - 3s 343ms/step - loss: 0.0097 - binary_accuracy: 0.9948
0.5035
Epoch 1/10
10/10 [==============================] - 5s 390ms/step - loss: 0.1168 - binary_accuracy: 0.9170
Epoch 2/10
10/10 [==============================] - 4s 397ms/step - loss: 0.0685 - binary_accuracy: 0.9469
Epoch 3/10
10/10 [==============================] - 4s 399ms/step - loss: 0.0403 - binary_accuracy: 0.9698
Epoch 4/10
10/10 [==============================] - 4s 395ms/step - loss: 0.0235 - binary_accuracy: 0.9834
Epoch 5/10
10/10 [==============================] - 4s 397ms/step - loss: 0.0185 - binary_accuracy: 0.9881
Epoch 6/10
10/10 [==============================] - 4s 392ms/step - loss: 0.0137 - binary_accuracy: 0.9905
Epoch 7/10
10/10 [==============================] - 4s 411ms/step - loss: 0.0118 - binary_accuracy: 0.9931
Epoch 8/10
10/10 [==============================] - 4s 406ms/step - loss: 0.0100 - binary_accuracy: 0.9936
Epoch 9/10
10/10 [==============================] - 4s 369ms/step - loss: 0.0071 - binary_accuracy: 0.9965
Epoch 10/10
10/10 [==============================] - 3s 351ms/step - loss: 0.0066 - binary_accuracy: 0.9966
0.5018
Epoch 1/10
10/10 [==============================] - 5s 440ms/step - loss: 0.1069 - binary_accuracy: 0.9280
Epoch 2/10
10/10 [==============================] - 4s 433ms/step - loss: 0.0573 - binary_accuracy: 0.9561
Epoch 3/10
10/10 [==============================] - 4s 425ms/step - loss: 0.0347 - binary_accuracy: 0.9747
Epoch 4/10
10/10 [==============================] - 4s 425ms/step - loss: 0.0199 - binary_accuracy: 0.9849
Epoch 5/10
10/10 [==============================] - 4s 426ms/step - loss: 0.0144 - binary_accuracy: 0.9907
Epoch 6/10
10/10 [==============================] - 4s 412ms/step - loss: 0.0106 - binary_accuracy: 0.9930
Epoch 7/10
10/10 [==============================] - 4s 421ms/step - loss: 0.0109 - binary_accuracy: 0.9933
Epoch 8/10
10/10 [==============================] - 4s 441ms/step - loss: 0.0072 - binary_accuracy: 0.9955
Epoch 9/10
10/10 [==============================] - 4s 414ms/step - loss: 0.0069 - binary_accuracy: 0.9966
Epoch 10/10
10/10 [==============================] - 4s 388ms/step - loss: 0.0057 - binary_accuracy: 0.9966
0.5029
Epoch 1/10
10/10 [==============================] - 5s 386ms/step - loss: 0.0964 - binary_accuracy: 0.9333
Epoch 2/10
10/10 [==============================] - 5s 460ms/step - loss: 0.0554 - binary_accuracy: 0.9580
Epoch 3/10
10/10 [==============================] - 5s 461ms/step - loss: 0.0291 - binary_accuracy: 0.9788
Epoch 4/10
10/10 [==============================] - 4s 435ms/step - loss: 0.0208 - binary_accuracy: 0.9848
Epoch 5/10
10/10 [==============================] - 4s 435ms/step - loss: 0.0143 - binary_accuracy: 0.9909
Epoch 6/10
10/10 [==============================] - 4s 444ms/step - loss: 0.0121 - binary_accuracy: 0.9925
Epoch 7/10
10/10 [==============================] - 4s 439ms/step - loss: 0.0104 - binary_accuracy: 0.9934
Epoch 8/10
10/10 [==============================] - 5s 469ms/step - loss: 0.0074 - binary_accuracy: 0.9962
Epoch 9/10
10/10 [==============================] - 4s 443ms/step - loss: 0.0060 - binary_accuracy: 0.9964
Epoch 10/10
10/10 [==============================] - 4s 410ms/step - loss: 0.0052 - binary_accuracy: 0.9971
0.496
Epoch 1/10
10/10 [==============================] - 30s 509ms/step - loss: 0.0920 - binary_accuracy: 0.9384
Epoch 2/10
10/10 [==============================] - 4s 433ms/step - loss: 0.0458 - binary_accuracy: 0.9651
Epoch 3/10
10/10 [==============================] - 4s 418ms/step - loss: 0.0273 - binary_accuracy: 0.9802
Epoch 4/10
10/10 [==============================] - 4s 408ms/step - loss: 0.0166 - binary_accuracy: 0.9885
Epoch 5/10
10/10 [==============================] - 4s 423ms/step - loss: 0.0132 - binary_accuracy: 0.9904
Epoch 6/10
10/10 [==============================] - 4s 411ms/step - loss: 0.0085 - binary_accuracy: 0.9950
Epoch 7/10
10/10 [==============================] - 4s 426ms/step - loss: 0.0086 - binary_accuracy: 0.9939
Epoch 8/10
10/10 [==============================] - 4s 424ms/step - loss: 0.0072 - binary_accuracy: 0.9952
Epoch 9/10
10/10 [==============================] - 4s 410ms/step - loss: 0.0057 - binary_accuracy: 0.9965
Epoch 10/10
10/10 [==============================] - 4s 394ms/step - loss: 0.0047 - binary_accuracy: 0.9969
0.4979
Epoch 1/10
10/10 [==============================] - 7s 663ms/step - loss: 0.0691 - binary_accuracy: 0.9508
Epoch 2/10
10/10 [==============================] - 5s 463ms/step - loss: 0.0417 - binary_accuracy: 0.9675
Epoch 3/10
10/10 [==============================] - 4s 423ms/step - loss: 0.0260 - binary_accuracy: 0.9799
Epoch 4/10
10/10 [==============================] - 4s 406ms/step - loss: 0.0169 - binary_accuracy: 0.9890
Epoch 5/10
10/10 [==============================] - 4s 393ms/step - loss: 0.0096 - binary_accuracy: 0.9937
Epoch 6/10
10/10 [==============================] - 4s 400ms/step - loss: 0.0075 - binary_accuracy: 0.9951
Epoch 7/10
10/10 [==============================] - 4s 401ms/step - loss: 0.0061 - binary_accuracy: 0.9961
Epoch 8/10
10/10 [==============================] - 4s 400ms/step - loss: 0.0044 - binary_accuracy: 0.9978
Epoch 9/10
10/10 [==============================] - 4s 374ms/step - loss: 0.0043 - binary_accuracy: 0.9975
Epoch 10/10
10/10 [==============================] - 4s 346ms/step - loss: 0.0033 - binary_accuracy: 0.9981
Fake Image Accuracy: 0.9635
Real Image Accuracy: 0.9593
True Positives (Fake): 9635
False Positives: 407
False Negatives: 365
True Negatives (Real): 9593