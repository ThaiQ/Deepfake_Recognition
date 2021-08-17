Model: "DeepfakeDetector"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 192, 256, 3)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 93, 125, 16)       2368
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 93, 125, 32)       4640
_________________________________________________________________
batch_normalization (BatchNo (None, 93, 125, 32)       128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 46, 62, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 46, 62, 32)        9248
_________________________________________________________________
batch_normalization_1 (Batch (None, 46, 62, 32)        128
_________________________________________________________________
dropout (Dropout)            (None, 46, 62, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 46, 62, 64)        18496
_________________________________________________________________
batch_normalization_2 (Batch (None, 46, 62, 64)        256
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 23, 31, 64)        36928
_________________________________________________________________
batch_normalization_3 (Batch (None, 23, 31, 64)        256
_________________________________________________________________
dropout_1 (Dropout)          (None, 23, 31, 64)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 16, 128)       73856
_________________________________________________________________
batch_normalization_4 (Batch (None, 12, 16, 128)       512
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 16, 128)       0
_________________________________________________________________
flatten (Flatten)            (None, 24576)             0
_________________________________________________________________
dense (Dense)                (None, 1024)              25166848
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 1025
=================================================================
Total params: 25,314,689
Trainable params: 25,314,049
Non-trainable params: 640
_________________________________________________________________
0.0916
2021-08-12 22:03:52.671051: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/10
2021-08-12 22:03:53.102811: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-08-12 22:03:53.576179: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-08-12 22:03:54.274800: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-08-12 22:03:55.602051: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2021-08-12 22:03:55.631849: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2021-08-12 22:03:58.149724: I tensorflow/stream_executor/cuda/cuda_blas.cc:1838] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
5/5 [==============================] - 10s 805ms/step - loss: 0.8110 - binary_accuracy: 0.5454 - val_loss: 3.1129 - val_binary_accuracy: 0.1290
Epoch 2/10
5/5 [==============================] - 2s 474ms/step - loss: 0.2131 - binary_accuracy: 0.3834 - val_loss: 0.6772 - val_binary_accuracy: 0.8920
Epoch 3/10
5/5 [==============================] - 2s 469ms/step - loss: 0.1190 - binary_accuracy: 0.8250 - val_loss: 0.6951 - val_binary_accuracy: 0.8500
Epoch 4/10
5/5 [==============================] - 2s 465ms/step - loss: 0.1204 - binary_accuracy: 0.8114 - val_loss: 0.6986 - val_binary_accuracy: 0.8810
Epoch 5/10
5/5 [==============================] - 2s 469ms/step - loss: 0.1137 - binary_accuracy: 0.8426 - val_loss: 0.6564 - val_binary_accuracy: 0.8770
Epoch 6/10
5/5 [==============================] - 2s 470ms/step - loss: 0.1121 - binary_accuracy: 0.8672 - val_loss: 0.6092 - val_binary_accuracy: 0.8100
Epoch 7/10
5/5 [==============================] - 2s 471ms/step - loss: 0.1101 - binary_accuracy: 0.7826 - val_loss: 0.6702 - val_binary_accuracy: 0.5460
Epoch 8/10
5/5 [==============================] - 2s 505ms/step - loss: 0.1116 - binary_accuracy: 0.5181 - val_loss: 0.6251 - val_binary_accuracy: 0.6570
Epoch 9/10
5/5 [==============================] - 2s 471ms/step - loss: 0.1089 - binary_accuracy: 0.7014 - val_loss: 0.6044 - val_binary_accuracy: 0.7190
Epoch 10/10
5/5 [==============================] - 2s 484ms/step - loss: 0.1064 - binary_accuracy: 0.6986 - val_loss: 0.5808 - val_binary_accuracy: 0.7280
0.0953
Epoch 1/10
5/5 [==============================] - 4s 582ms/step - loss: 0.1144 - binary_accuracy: 0.6488 - val_loss: 0.6911 - val_binary_accuracy: 0.5120
Epoch 2/10
5/5 [==============================] - 2s 435ms/step - loss: 0.1105 - binary_accuracy: 0.6336 - val_loss: 0.5601 - val_binary_accuracy: 0.7180
Epoch 3/10
5/5 [==============================] - 2s 465ms/step - loss: 0.1091 - binary_accuracy: 0.6426 - val_loss: 0.6101 - val_binary_accuracy: 0.6370
Epoch 4/10
5/5 [==============================] - 2s 466ms/step - loss: 0.1040 - binary_accuracy: 0.6890 - val_loss: 0.5553 - val_binary_accuracy: 0.7020
Epoch 5/10
5/5 [==============================] - 2s 475ms/step - loss: 0.1074 - binary_accuracy: 0.6430 - val_loss: 0.6723 - val_binary_accuracy: 0.5410
Epoch 6/10
5/5 [==============================] - 2s 469ms/step - loss: 0.1004 - binary_accuracy: 0.6958 - val_loss: 0.5749 - val_binary_accuracy: 0.6840
Epoch 7/10
5/5 [==============================] - 2s 468ms/step - loss: 0.0990 - binary_accuracy: 0.6826 - val_loss: 0.6391 - val_binary_accuracy: 0.6190
Epoch 8/10
5/5 [==============================] - 2s 467ms/step - loss: 0.0940 - binary_accuracy: 0.7306 - val_loss: 0.4637 - val_binary_accuracy: 0.7740
Epoch 9/10
5/5 [==============================] - 2s 464ms/step - loss: 0.0921 - binary_accuracy: 0.7446 - val_loss: 0.6455 - val_binary_accuracy: 0.6300
Epoch 10/10
5/5 [==============================] - 2s 469ms/step - loss: 0.0894 - binary_accuracy: 0.7462 - val_loss: 0.6068 - val_binary_accuracy: 0.6540
0.0997
Epoch 1/10
5/5 [==============================] - 7s 504ms/step - loss: 0.1068 - binary_accuracy: 0.6500 - val_loss: 0.5544 - val_binary_accuracy: 0.6960
Epoch 2/10
5/5 [==============================] - 2s 424ms/step - loss: 0.1060 - binary_accuracy: 0.6738 - val_loss: 0.5720 - val_binary_accuracy: 0.6620
Epoch 3/10
5/5 [==============================] - 2s 469ms/step - loss: 0.1026 - binary_accuracy: 0.6986 - val_loss: 0.5872 - val_binary_accuracy: 0.6630
Epoch 4/10
5/5 [==============================] - 2s 475ms/step - loss: 0.0950 - binary_accuracy: 0.7014 - val_loss: 0.4446 - val_binary_accuracy: 0.7870
Epoch 5/10
5/5 [==============================] - 2s 473ms/step - loss: 0.0991 - binary_accuracy: 0.6812 - val_loss: 0.4443 - val_binary_accuracy: 0.7940
Epoch 6/10
5/5 [==============================] - 2s 478ms/step - loss: 0.1021 - binary_accuracy: 0.7304 - val_loss: 0.7855 - val_binary_accuracy: 0.4740
Epoch 7/10
5/5 [==============================] - 2s 485ms/step - loss: 0.0965 - binary_accuracy: 0.7028 - val_loss: 0.6227 - val_binary_accuracy: 0.6310
Epoch 8/10
5/5 [==============================] - 2s 476ms/step - loss: 0.0892 - binary_accuracy: 0.7072 - val_loss: 0.4033 - val_binary_accuracy: 0.8060
Epoch 9/10
5/5 [==============================] - 2s 472ms/step - loss: 0.0854 - binary_accuracy: 0.7610 - val_loss: 0.5065 - val_binary_accuracy: 0.7380
Epoch 10/10
5/5 [==============================] - 2s 475ms/step - loss: 0.0783 - binary_accuracy: 0.7912 - val_loss: 0.5085 - val_binary_accuracy: 0.7260
0.0955
Epoch 1/10
5/5 [==============================] - 4s 594ms/step - loss: 0.0982 - binary_accuracy: 0.6952 - val_loss: 0.6401 - val_binary_accuracy: 0.6620
Epoch 2/10
5/5 [==============================] - 2s 452ms/step - loss: 0.0928 - binary_accuracy: 0.7372 - val_loss: 0.4900 - val_binary_accuracy: 0.7650
Epoch 3/10
5/5 [==============================] - 2s 463ms/step - loss: 0.0895 - binary_accuracy: 0.7654 - val_loss: 0.5516 - val_binary_accuracy: 0.6870
Epoch 4/10
5/5 [==============================] - 2s 459ms/step - loss: 0.0873 - binary_accuracy: 0.7434 - val_loss: 0.5979 - val_binary_accuracy: 0.6880
Epoch 5/10
5/5 [==============================] - 2s 466ms/step - loss: 0.0776 - binary_accuracy: 0.7840 - val_loss: 0.5646 - val_binary_accuracy: 0.7130
Epoch 6/10
5/5 [==============================] - 2s 465ms/step - loss: 0.0778 - binary_accuracy: 0.7678 - val_loss: 0.3906 - val_binary_accuracy: 0.8260
Epoch 7/10
5/5 [==============================] - 2s 461ms/step - loss: 0.0747 - binary_accuracy: 0.7872 - val_loss: 0.5317 - val_binary_accuracy: 0.7400
Epoch 8/10
5/5 [==============================] - 2s 454ms/step - loss: 0.0731 - binary_accuracy: 0.7998 - val_loss: 0.4570 - val_binary_accuracy: 0.7840
Epoch 9/10
5/5 [==============================] - 2s 458ms/step - loss: 0.0647 - binary_accuracy: 0.8186 - val_loss: 0.3232 - val_binary_accuracy: 0.8610
Epoch 10/10
5/5 [==============================] - 2s 457ms/step - loss: 0.0648 - binary_accuracy: 0.8428 - val_loss: 0.4123 - val_binary_accuracy: 0.8090
0.0991
Epoch 1/10
5/5 [==============================] - 3s 422ms/step - loss: 0.0929 - binary_accuracy: 0.7604 - val_loss: 0.4012 - val_binary_accuracy: 0.8100
Epoch 2/10
5/5 [==============================] - 2s 483ms/step - loss: 0.0922 - binary_accuracy: 0.7204 - val_loss: 0.4139 - val_binary_accuracy: 0.8000
Epoch 3/10
5/5 [==============================] - 2s 495ms/step - loss: 0.0824 - binary_accuracy: 0.7860 - val_loss: 0.4557 - val_binary_accuracy: 0.7990
Epoch 4/10
5/5 [==============================] - 2s 477ms/step - loss: 0.0803 - binary_accuracy: 0.7982 - val_loss: 0.3076 - val_binary_accuracy: 0.8730
Epoch 5/10
5/5 [==============================] - 2s 471ms/step - loss: 0.0718 - binary_accuracy: 0.8284 - val_loss: 0.3133 - val_binary_accuracy: 0.8590
Epoch 6/10
5/5 [==============================] - 2s 479ms/step - loss: 0.0626 - binary_accuracy: 0.8412 - val_loss: 0.3103 - val_binary_accuracy: 0.8700
Epoch 7/10
5/5 [==============================] - 2s 479ms/step - loss: 0.0598 - binary_accuracy: 0.8568 - val_loss: 0.3530 - val_binary_accuracy: 0.8530
Epoch 8/10
5/5 [==============================] - 2s 474ms/step - loss: 0.0543 - binary_accuracy: 0.8646 - val_loss: 0.2663 - val_binary_accuracy: 0.9010
Epoch 9/10
5/5 [==============================] - 2s 508ms/step - loss: 0.0530 - binary_accuracy: 0.8652 - val_loss: 0.2675 - val_binary_accuracy: 0.8950
Epoch 10/10
5/5 [==============================] - 2s 477ms/step - loss: 0.0456 - binary_accuracy: 0.8872 - val_loss: 0.2870 - val_binary_accuracy: 0.8850
0.0942
Epoch 1/10
5/5 [==============================] - 9s 2s/step - loss: 0.0812 - binary_accuracy: 0.7900 - val_loss: 0.3524 - val_binary_accuracy: 0.8560
Epoch 2/10
5/5 [==============================] - 2s 460ms/step - loss: 0.0781 - binary_accuracy: 0.8084 - val_loss: 0.4019 - val_binary_accuracy: 0.8190
Epoch 3/10
5/5 [==============================] - 3s 512ms/step - loss: 0.0716 - binary_accuracy: 0.8110 - val_loss: 0.3169 - val_binary_accuracy: 0.8690
Epoch 4/10
5/5 [==============================] - 2s 482ms/step - loss: 0.0626 - binary_accuracy: 0.8322 - val_loss: 0.2583 - val_binary_accuracy: 0.8960
Epoch 5/10
5/5 [==============================] - 2s 466ms/step - loss: 0.0545 - binary_accuracy: 0.8694 - val_loss: 0.3065 - val_binary_accuracy: 0.8650
Epoch 6/10
5/5 [==============================] - 2s 466ms/step - loss: 0.0531 - binary_accuracy: 0.8694 - val_loss: 0.2961 - val_binary_accuracy: 0.8750
Epoch 7/10
5/5 [==============================] - 2s 508ms/step - loss: 0.0486 - binary_accuracy: 0.8864 - val_loss: 0.3513 - val_binary_accuracy: 0.8450
Epoch 8/10
5/5 [==============================] - 2s 480ms/step - loss: 0.0421 - binary_accuracy: 0.9004 - val_loss: 0.2583 - val_binary_accuracy: 0.8900
Epoch 9/10
5/5 [==============================] - 2s 475ms/step - loss: 0.0420 - binary_accuracy: 0.9038 - val_loss: 0.3321 - val_binary_accuracy: 0.8500
Epoch 10/10
5/5 [==============================] - 2s 475ms/step - loss: 0.0334 - binary_accuracy: 0.9292 - val_loss: 0.2818 - val_binary_accuracy: 0.8780
0.0995
Epoch 1/10
5/5 [==============================] - 11s 843ms/step - loss: 0.0778 - binary_accuracy: 0.8138 - val_loss: 0.3127 - val_binary_accuracy: 0.8760
Epoch 2/10
5/5 [==============================] - 2s 450ms/step - loss: 0.0711 - binary_accuracy: 0.8480 - val_loss: 0.3851 - val_binary_accuracy: 0.8310
Epoch 3/10
5/5 [==============================] - 2s 489ms/step - loss: 0.0592 - binary_accuracy: 0.8598 - val_loss: 0.3225 - val_binary_accuracy: 0.8720
Epoch 4/10
5/5 [==============================] - 2s 475ms/step - loss: 0.0519 - binary_accuracy: 0.8734 - val_loss: 0.2702 - val_binary_accuracy: 0.8830
Epoch 5/10
5/5 [==============================] - 2s 474ms/step - loss: 0.0454 - binary_accuracy: 0.8960 - val_loss: 0.2100 - val_binary_accuracy: 0.9110
Epoch 6/10
5/5 [==============================] - 2s 443ms/step - loss: 0.0425 - binary_accuracy: 0.9040 - val_loss: 0.1944 - val_binary_accuracy: 0.9330
Epoch 7/10
5/5 [==============================] - 2s 484ms/step - loss: 0.0390 - binary_accuracy: 0.9114 - val_loss: 0.1996 - val_binary_accuracy: 0.9300
Epoch 8/10
5/5 [==============================] - 2s 479ms/step - loss: 0.0418 - binary_accuracy: 0.9138 - val_loss: 0.2235 - val_binary_accuracy: 0.9060
Epoch 9/10
5/5 [==============================] - 2s 480ms/step - loss: 0.0353 - binary_accuracy: 0.9152 - val_loss: 0.2140 - val_binary_accuracy: 0.9290
Epoch 10/10
5/5 [==============================] - 2s 483ms/step - loss: 0.0289 - binary_accuracy: 0.9344 - val_loss: 0.2297 - val_binary_accuracy: 0.9130
0.0965
Epoch 1/10
5/5 [==============================] - 3s 461ms/step - loss: 0.0682 - binary_accuracy: 0.8560 - val_loss: 0.3618 - val_binary_accuracy: 0.8460
Epoch 2/10
5/5 [==============================] - 2s 440ms/step - loss: 0.0679 - binary_accuracy: 0.8370 - val_loss: 0.3282 - val_binary_accuracy: 0.8600
Epoch 3/10
5/5 [==============================] - 2s 474ms/step - loss: 0.0571 - binary_accuracy: 0.8528 - val_loss: 0.2923 - val_binary_accuracy: 0.8800
Epoch 4/10
5/5 [==============================] - 2s 478ms/step - loss: 0.0450 - binary_accuracy: 0.8906 - val_loss: 0.2131 - val_binary_accuracy: 0.9160
Epoch 5/10
5/5 [==============================] - 2s 472ms/step - loss: 0.0365 - binary_accuracy: 0.9132 - val_loss: 0.1926 - val_binary_accuracy: 0.9180
Epoch 6/10
5/5 [==============================] - 2s 475ms/step - loss: 0.0375 - binary_accuracy: 0.9098 - val_loss: 0.1492 - val_binary_accuracy: 0.9430
Epoch 7/10
5/5 [==============================] - 2s 481ms/step - loss: 0.0408 - binary_accuracy: 0.8918 - val_loss: 0.1535 - val_binary_accuracy: 0.9390
Epoch 8/10
5/5 [==============================] - 2s 471ms/step - loss: 0.0357 - binary_accuracy: 0.9142 - val_loss: 0.1503 - val_binary_accuracy: 0.9470
Epoch 9/10
5/5 [==============================] - 2s 477ms/step - loss: 0.0382 - binary_accuracy: 0.9180 - val_loss: 0.2235 - val_binary_accuracy: 0.9050
Epoch 10/10
5/5 [==============================] - 2s 476ms/step - loss: 0.0295 - binary_accuracy: 0.9420 - val_loss: 0.2826 - val_binary_accuracy: 0.8840
0.0965
Epoch 1/10
5/5 [==============================] - 5s 907ms/step - loss: 0.0579 - binary_accuracy: 0.8630 - val_loss: 0.2469 - val_binary_accuracy: 0.9020
Epoch 2/10
5/5 [==============================] - 2s 513ms/step - loss: 0.0503 - binary_accuracy: 0.8802 - val_loss: 0.2682 - val_binary_accuracy: 0.8920
Epoch 3/10
5/5 [==============================] - 2s 518ms/step - loss: 0.0434 - binary_accuracy: 0.8984 - val_loss: 0.2199 - val_binary_accuracy: 0.8960
Epoch 4/10
5/5 [==============================] - 3s 523ms/step - loss: 0.0373 - binary_accuracy: 0.9124 - val_loss: 0.1363 - val_binary_accuracy: 0.9500
Epoch 5/10
5/5 [==============================] - 2s 497ms/step - loss: 0.0332 - binary_accuracy: 0.9170 - val_loss: 0.1226 - val_binary_accuracy: 0.9580
Epoch 6/10
5/5 [==============================] - 2s 503ms/step - loss: 0.0347 - binary_accuracy: 0.9182 - val_loss: 0.1275 - val_binary_accuracy: 0.9540
Epoch 7/10
5/5 [==============================] - 3s 534ms/step - loss: 0.0283 - binary_accuracy: 0.9380 - val_loss: 0.1386 - val_binary_accuracy: 0.9490
Epoch 8/10
5/5 [==============================] - 2s 495ms/step - loss: 0.0259 - binary_accuracy: 0.9384 - val_loss: 0.1215 - val_binary_accuracy: 0.9530
Epoch 9/10
5/5 [==============================] - 2s 489ms/step - loss: 0.0221 - binary_accuracy: 0.9576 - val_loss: 0.1673 - val_binary_accuracy: 0.9360
Epoch 10/10
5/5 [==============================] - 2s 479ms/step - loss: 0.0227 - binary_accuracy: 0.9466 - val_loss: 0.1527 - val_binary_accuracy: 0.9390
0.098
Epoch 1/10
5/5 [==============================] - 5s 926ms/step - loss: 0.0639 - binary_accuracy: 0.8786 - val_loss: 0.2043 - val_binary_accuracy: 0.9120
Epoch 2/10
5/5 [==============================] - 2s 425ms/step - loss: 0.0545 - binary_accuracy: 0.8692 - val_loss: 0.1590 - val_binary_accuracy: 0.9400
Epoch 3/10
5/5 [==============================] - 2s 462ms/step - loss: 0.0435 - binary_accuracy: 0.8910 - val_loss: 0.2484 - val_binary_accuracy: 0.8930
Epoch 4/10
5/5 [==============================] - 2s 477ms/step - loss: 0.0387 - binary_accuracy: 0.8992 - val_loss: 0.1776 - val_binary_accuracy: 0.9410
Epoch 5/10
5/5 [==============================] - 2s 470ms/step - loss: 0.0365 - binary_accuracy: 0.9272 - val_loss: 0.2589 - val_binary_accuracy: 0.8940
Epoch 6/10
5/5 [==============================] - 2s 469ms/step - loss: 0.0295 - binary_accuracy: 0.9336 - val_loss: 0.1783 - val_binary_accuracy: 0.9300
5/5 [==============================] - 2s 466ms/step - loss: 0.0270 - binary_accuracy: 0.9384 - val_loss: 0.1480 - val_binary_accuracy: 0.9460
Epoch 8/10
5/5 [==============================] - 2s 470ms/step - loss: 0.0252 - binary_accuracy: 0.9432 - val_loss: 0.1321 - val_binary_accuracy: 0.9560
Epoch 9/10
5/5 [==============================] - 2s 483ms/step - loss: 0.0235 - binary_accuracy: 0.9522 - val_loss: 0.2146 - val_binary_accuracy: 0.9180
Epoch 10/10
5/5 [==============================] - 2s 476ms/step - loss: 0.0179 - binary_accuracy: 0.9628 - val_loss: 0.1514 - val_binary_accuracy: 0.9390
True Positives (Fake): 1292
False Positives: 43
False Negatives: 216
True Negatives (Real): 89
Fake Image Accuracy: 0.8567639257294429
Real Image Accuracy: 0.6742424242424242
Done