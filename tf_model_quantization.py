# import logging
# logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
import numpy as np
import wandb
import tempfile
import zipfile
import os
import pathlib
from tf_get_data import get_dataset



#This function is used to calculate the size of the model
def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)




def convert(model, train_data, main_folder):
    print("Converting and quantising model to tflite")

    ####DEFINE REPRESENTATIVE DATASET####
    def representative_data_gen_():
        X_train = tf.convert_to_tensor(np.array([data.values for data in train_data]).astype("float32")) #NOTE: I am unsure about whether I should convert the tensor to int8 already in the representative dataset
        for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(400):
            yield [input_value]


    ####GET QUANTISED MODEL###
    
    #Get the representative dataset generator
    representative_data_gen = representative_data_gen_

    # create the converter object
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # set the optimisations (this means that we can in theory tell the model to optimise for size latency or sparsity but optimisation for size and latency are deprecated while optimisation for sparsity is experimental. In the documentation the suggested option is DEFAULT)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    # set the representative dataset for converter (needed for full-integer quantization)
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to int8 
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Get the quantized model
    tflite_model_quant = converter.convert()
    

    #Create temporary file to calculate size of quantized model
    _, tflite_file_to_zip = tempfile.mkstemp('.tflite')
    with open(tflite_file_to_zip, 'wb') as f:
        f.write(tflite_model_quant)
    wandb.log({"model_size [KB]": get_gzipped_model_size(tflite_file_to_zip)//1000})


    ####SAVE THE MODEL####


    # Save the quantized model:
    tflite_model_file = pathlib.Path(main_folder) / "tf_model_quantized.tflite"
    tflite_model_file.write_bytes(tflite_model_quant)

    return tflite_model_quant

