#!/bin/bash

python3  ./model_run.py \
 --model ./Model_Openvino_IR/MNIST_CNN_IR/MNIST_CNN_IR.xml \
 --xdata ./Data_Test/test_data_MNIST_CNN.npy \
 --ydata ./Data_Test/test_data_y_MNIST_CNN.npy \
 --outdir ./Results \
 --title 'Accelerated with intel NCS2'

