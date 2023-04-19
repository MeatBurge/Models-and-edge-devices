###############################################################################################
# The primary function of this program is to test the model's performance, load and run the 
# TensorFlow Lite or OpenVino IR model, read the specified test dataset file and save the test 
# results to a data file in json format.
#
# Usage:
#      > python3  model_run.py 
#        --model  Model filename (tflite or xml file format) 
#        --xdata  X data file (data file for model inference input) 
#        --ydata  y data file (labels or values corresponding to the data file)
#        --outdir Path to save results
#        --title  model name
#
#   File: model_run.py
#   Author: Yixiao Yuan
#   Date: Apr 10, 2023
#   Purpose: CSCI 6709, Project
###########################################################################################

import os
import time
import json
import psutil
import threading
import numpy as np
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
import openvino.inference_engine as IECore
from sys import argv
from datetime import datetime

results_data ={}


# The function sets a global dictionary "results_data" with a given name and data.
def set_results_data(name, data):
    global results_data
    results_data [name]  = data
    return


# This function calculates the elapsed time in milliseconds between two given datetime objects.
def calc_elapsed_time(start_time, end_time):
    return int((end_time-start_time).total_seconds() * 1000)

    
# This function monitors the system usage (CPU and memory) of a given process over time.
def monitor_sys_usage(begin_time, process):
    etime,mem,cpu = [],[],[]
    while not stop_monitoring:
        elapsed_time = calc_elapsed_time(begin_time, datetime.now())
        # Get the memory usage of the process
        memory_info = process.memory_info()
        mem_usage = round((memory_info.rss + memory_info.vms)/(1024*1024),2)
        cpu_usage = psutil.cpu_percent(interval=0.5)
        
        etime.append(elapsed_time)
        mem.append(mem_usage)
        cpu.append(cpu_usage)
        
        time.sleep(0.5)
    
    set_results_data('elapsed time', etime)
    set_results_data('mem usage', mem)
    set_results_data('cpu usage', cpu)


# Get the memory usage of the process
def get_memory_used(process):
    # Get the memory usage of the process
    memory_info = process.memory_info()
    mem_used = round((memory_info.rss + memory_info.vms)/(1024*1024),2)
    # Print the memory usage information
    print("Memory used: {:.2f} M".format(mem_used))

    return mem_used


# Get elapsed time between the start and end times.
def get_elapsed_time(start_time):
    # the current time as the end time for measuring the elapsed time
    end_time = datetime.now()

    # calculates the elapsed time between the start and end times in seconds.
    # and then multiplies it by 1000 to convert it to milliseconds.
    elapsed_time = calc_elapsed_time(start_time, end_time)
    print("Elapsed time: {:.2f} ms".format(elapsed_time))

    return elapsed_time


# This function returns the elapsed time and memory used by a process given the start time.
def get_elapsed_time_memory_used(start_time, process):
    elapsed_time = get_elapsed_time(start_time)
    mem_used = get_memory_used(process)
    return elapsed_time, mem_used


# loads a TensorFlow Lite model and returns the interpreter object, input details, and output details.
def load_tflite_model(model_file):
    # interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    # Get input and output tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


# This function loads an OpenVINO IR model and returns the IECore object, input and output format.
def load_ir_model(model_file):
    model_xml = model_file
    model_bin = os.path.splitext(model_file)[0]+".bin"

    # Load the openvino IR model
    ie = IECore.IECore()
    ie_net = ie.read_network(model=model_xml, weights=model_bin)
    #  ie_exec = ie.load_network(network=ie_net, device_name="CPU")
    ie_exec = ie.load_network(network=ie_net, device_name="MYRIAD")
    # get input and output format 
    input_blob = next(iter(ie_net.input_info))
    output_blob = next(iter(ie_net.outputs))
    return ie_exec, input_blob, output_blob


#  Run the Tensorflow lite and the OpenVino IR model
def run_Model(begin_time, process, model_file, xdata_file, ydata_file, model_type):
    print("\n>> Start...")
    # Get the memory usage of the process
    get_memory_used(process)

    time_points = []
    
    # measuring the elapsed time
    start_time = datetime.now()
    elapsed_time = calc_elapsed_time(begin_time, start_time)
    time_points.append({"start": elapsed_time})
    
    # Load the tflite model
    print("\n>> Load {} ...".format(model_file))
    if model_type == ".tflite" :
        interpreter, input_details, output_details = load_tflite_model(model_file)
        
    # Load the IR model and create the inference engine
    if model_type == ".xml" :
        ie_exec, input_blob, output_blob = load_ir_model(model_file)

    # the elapsed time and the memory usage of the process
    load_time,load_mem = get_elapsed_time_memory_used(start_time, process)

    # measuring the elapsed time
    start_time = datetime.now()
    elapsed_time = calc_elapsed_time(begin_time, start_time)
    time_points.append({"load model": elapsed_time})
    
    # load x test data
    print("\n>> Load {}...".format(xdata_file))
    x_test_data = np.load(xdata_file, mmap_mode='r')

    # the elapsed time and the memory usage of the process
    get_elapsed_time_memory_used(start_time, process)

    # measuring the elapsed time
    start_time = datetime.now()
    elapsed_time = calc_elapsed_time(begin_time, start_time)
    time_points.append({"load data":elapsed_time})
    
    # Start inference process
    print("\n>> Run inference process...")
    progress = 0
    predictions = []

    # Load data in chunks
    data_len = len(x_test_data)
    chunk_size = 100
    num_chunks = data_len // chunk_size
    for j in range(num_chunks):
        x_test_data_chunk = x_test_data[j*chunk_size:(j+1)*chunk_size]

        # Process data chunk
        for i in range(len(x_test_data_chunk)):
            # Print complete percentage
            new_progress = (100*(j * chunk_size + i)/data_len + 0.1)
            if (new_progress - progress >= 1):
                progress = new_progress
                print("\rProgress: {}% completed.".format(int(progress)), end="")

            if model_type == ".tflite" :
                # Ready to input data
                input_data = np.expand_dims(x_test_data_chunk[i], axis=0).astype(
                    input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], input_data)
                # Run a reasoning
                interpreter.invoke()
                # Get output data
                output_data = interpreter.get_tensor(output_details[0]['index'])

            if model_type == ".xml" :
                # Ready to input data
                input_data= np.expand_dims(x_test_data_chunk[i],0)
                input_data= input_data.transpose((0, 3, 1, 2))

                # Run a reasoning
                res = ie_exec.infer(inputs={input_blob: input_data})
                output_data = res[output_blob]
            
            # Get output data
            predictions.append(output_data)

    print()
    # the elapsed time and the memory usage of the process
    infer_time,infer_mem = get_elapsed_time_memory_used(start_time, process)

    # measuring the elapsed time
    start_time = datetime.now()
    elapsed_time = calc_elapsed_time(begin_time, start_time)
    time_points.append({"inference time":elapsed_time})
    
    # Convert the prediction result to the Numpy array
    predictions = np.array(predictions)
    predictions = predictions.reshape(predictions.shape[0], 10)

    if type(predictions[0][0]) == np.uint8:
        y_pred_lite = (predictions >= 128).astype(int)
    else:
        y_pred_lite = (predictions > 0.5).astype(int)

    y_pred_lite = np.argmax(y_pred_lite, axis=1)

    y_test_lite = np.load(ydata_file)

    print("\n>> Calculate the inference accuracy...")
    # the elapsed time and the memory usage of the process
    eval_time,eval_mem = get_elapsed_time_memory_used(start_time, process)

    #calculating the accuracy 
    correct_predictions = np.sum(y_test_lite== y_pred_lite)
    acc = correct_predictions/len(y_test_lite )
    print("Accuracy: ", acc)

    # measuring the elapsed time
    start_time = datetime.now()
    elapsed_time = calc_elapsed_time(begin_time, start_time)
    time_points.append({"evaluation time" : elapsed_time})

    set_results_data("model loading time",load_time)
    set_results_data("inference time", infer_time)
    set_results_data("number of inferences",len(y_test_lite ))
    set_results_data("average inference time", infer_time/len(y_test_lite))
    set_results_data("memory usage", eval_mem)
    set_results_data("accuracy", acc)
    set_results_data("time points", time_points)


# Get the string specified in the command line
def get_args_label(flag):
    res = ""
    if flag in argv:
        if (argv.index(flag)+1 < len(argv)):
            res = argv[argv.index(flag)+1]
    return res


# This function gets command-line arguments and checks if they exist and are in the correct format,
# and returns them if they are valid.
#
# :return: a tuple containing the values of the command-line arguments: model_file, xdata_file,
# ydata_file, model_title, and out_file. If there is an error with the arguments, the
# function prints an error message and returns None.
def get_args_content():
    # Get command-line args
    model_file = get_args_label("--model")
    xdata_file = get_args_label("--xdata")
    ydata_file = get_args_label("--ydata")
    out_dir = get_args_label("--outdir")
    model_title = get_args_label("--title")

    args_isok = True
    if (args_isok and (not os.path.exists(model_file))):
        args_isok = False
        print('\nError:  Missing required argument "--model"  or model file does not exist.')
    if (args_isok and (not os.path.exists(xdata_file))):
        args_isok = False
        print('\nError:  Missing required argument "--xdata"  or X data file does not exist.')
    if (args_isok and (not os.path.exists(ydata_file))):
        args_isok = False
        print('\nError:  Missing required argument "--ydata"  or y data file does not exist.')

    if (args_isok):
        if out_dir == "":
            out_dir = "./"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if out_dir[-1] != "/":
            out_dir = out_dir +"/"
        if model_title == "":
            model_title = os.path.splitext(os.path.basename(model_file))[0]

    model_type = os.path.splitext(os.path.basename(model_file))[1].lower()
    if model_type not in ['.tflite', '.xml']:
        args_isok = False
        print('\nError:  The model only supports tensorflow lite and openvino ir formats, so the model file should have a .tflite or .xml suffix')
    if (args_isok):
            out_file = out_dir + os.path.splitext(os.path.basename(model_file))[0] + "_result.json"
            return (model_file, xdata_file, ydata_file, model_title, out_file)            
    else:
        print('\nUsage:')
        print('    > python3 model_run.py \\')
        print('      --model  model filename (tflite or xml file format) \\')
        print('      --xdata  X data file (data file for model inference input)\\')
        print('      --ydata  y data file (labels or values corresponding to the data file) \\')
        print('      --outdir Path to save results \\')
        print('      --title  model name')
        print('\nExample:')
        print('    > python3 model_run.py \\')
        print('      --model  iris.tflile \\')
        print('      --xdata  iris_x_data.npy \\')
        print('      --ydata  iris_y_data.npy \\')
        print('      --outdir ./results \\')
        print('      --title  "iris dataset classification model')
        
    return 


if __name__ == '__main__':
    
    # get command-line arguments 
    args_content = get_args_content()
    if(args_content == None):
        exit()
    
    model_file,xdata_file,ydata_file,model_title,out_file = args_content
    model_type = os.path.splitext(os.path.basename(model_file))[1].lower()

    # Get the process of the current Python program
    pid = os.getpid()
    process = psutil.Process(pid)
    
    begin_time = datetime.now()
    set_results_data("title", model_title)

    # Monitor CPU and memory usage
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_sys_usage, args=(begin_time, process))
    monitor_thread.start()
    
    model_thread = None  
    # model_type ='.tflite' or ".xml"
    model_thread = threading.Thread(target=run_Model,
                    args=(begin_time, process, model_file, xdata_file, ydata_file, model_type))
    model_thread.start()

    # Wait for model inference to complete
    model_thread.join()
    stop_monitoring = True
    monitor_thread.join()
    
    # save the resuls to json file
    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(results_data, file, ensure_ascii=False, indent=4)