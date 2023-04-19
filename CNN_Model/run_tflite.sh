#!/bin/bash

python3  ./model_run.py \
 --model ./Model_Tensorflow_Lite/MNIST_CNN.tflite \
 --xdata ./Data_Test/test_data_MNIST_CNN.npy \
 --ydata ./Data_Test/test_data_y_MNIST_CNN.npy \
 --outdir ./Results \
 --title 'No quantification'

python3  ./model_run.py \
 --model ./Model_Tensorflow_Lite/MNIST_CNN_quant_aware.tflite \
 --xdata ./Data_Test/test_data_MNIST_CNN.npy \
 --ydata ./Data_Test/test_data_y_MNIST_CNN.npy \
 --outdir ./Results \
 --title 'Quantization Aware Training'

python3  ./model_run.py \
 --model ./Model_Tensorflow_Lite/MNIST_CNN_quant.tflite \
 --xdata ./Data_Test/test_data_MNIST_CNN.npy \
 --ydata ./Data_Test/test_data_y_MNIST_CNN.npy \
 --outdir ./Results \
 --title 'Dynamic range quantization'

python3  ./model_run.py \
 --model ./Model_Tensorflow_Lite/MNIST_CNN_quant_float16.tflite \
 --xdata ./Data_Test/test_data_MNIST_CNN_float16.npy \
 --ydata ./Data_Test/test_data_y_MNIST_CNN.npy \
 --outdir ./Results \
 --title 'Float16 quantization'

python3  ./model_run.py \
 --model ./Model_Tensorflow_Lite/MNIST_CNN_quant_uint8.tflite \
 --xdata ./Data_Test/test_data_MNIST_CNN_uint8.npy \
 --ydata ./Data_Test/test_data_y_MNIST_CNN.npy \
 --outdir ./Results \
 --title 'Full integer quantization'

