Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 192, 256, 1)]     0           #Input layer, not much to say about it
_________________________________________________________________
random_flip (RandomFlip)     (None, 192, 256, 1)       0           #The 4 RandomX layers alter the input photo to increase model flexibility
_________________________________________________________________
random_rotation (RandomRotat (None, 192, 256, 1)       0
_________________________________________________________________
random_translation (RandomTr (None, 192, 256, 1)       0
_________________________________________________________________
random_zoom (RandomZoom)     (None, 192, 256, 1)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 96, 128, 32)       320         #Scales the pixels using kernels so that higher value means more important. Each one divides output size by 2
_________________________________________________________________ 
batch_normalization (BatchNo (None, 96, 128, 32)       128         #Normalizes pixel values
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 48, 64, 32)        0           #Drops useless pixels. Set stride to (2, 2), so each pooling layer reduces output size by half
_________________________________________________________________
spatial_dropout2d (SpatialDr (None, 48, 64, 32)        0           #Randomly drops kernels so as to reduce the model's reliance on them
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 32, 32)        9248
_________________________________________________________________
batch_normalization_1 (Batch (None, 24, 32, 32)        128
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 16, 32)        0
_________________________________________________________________
spatial_dropout2d_1 (Spatial (None, 12, 16, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 8, 64)          18496
_________________________________________________________________
batch_normalization_2 (Batch (None, 6, 8, 64)          256
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 4, 64)          0
_________________________________________________________________
spatial_dropout2d_2 (Spatial (None, 3, 4, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 768)               0
_________________________________________________________________
dense (Dense)                (None, 384)               295296
_________________________________________________________________
dropout (Dropout)            (None, 384)               0
_________________________________________________________________
dense_1 (Dense)              (None, 96)                36960
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 97
=================================================================
Total params: 360,929
Trainable params: 360,673
Non-trainable params: 256
