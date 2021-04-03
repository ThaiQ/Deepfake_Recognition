# Instruction
1. You need Python 3.7.6 (higher won't work)
2. Run `pip install -r dependencies-py3.7.6.txt`
3. Download VGG-16 and add to `SSD_Implement` folder

### Download the convolutionalized VGG-16 weights

In order to train an SSD300 or SSD512 from scratch, download the weights of the fully convolutionalized VGG-16 model trained to convergence on ImageNet classification here:

[`VGG_ILSVRC_16_layers_fc_reduced.h5`](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox).

As with all other weights files below, this is a direct port of the corresponding `.caffemodel` file that is provided in the repository of the original Caffe implementation.