# import logging
# logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
import numpy as np
import wandb
import tempfile
import zipfile
import os
import pathlib
import pandas as pd
from tf.tf_get_data import get_dataset
from typing import List



def get_gzipped_model_size(file: str) -> int:
    """
    Returns the size of the gzipped model, in bytes.

    :param file: Path to the model file
    :return: Size of the gzipped model in bytes
    """
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)




def convert(model: tf.keras.Model, train_data: List[pd.DataFrame], main_folder: str) -> bytes:
    """
    Converts and quantizes a TensorFlow model to TensorFlow Lite format.

    Args:
    - model: The TensorFlow model to be converted.
    - train_data: The training data used to define the representative dataset for quantization.
    - main_folder: The folder where the quantized model will be saved.

    Returns:
    - tflite_model_quant: The quantized TensorFlow Lite model.
    """
    print("Converting and quantizing model to tflite")

    # Define representative dataset generator
    def representative_data_gen():
        X_train = tf.convert_to_tensor(np.array([data.values for data in train_data]).astype("float32"))
        for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(400):
            yield [input_value]

    # Get the representative dataset generator
    representative_data_gen = representative_data_gen

    # Create the converter object
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Set the optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set the representative dataset for converter (needed for full-integer quantization)
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Get the quantized model
    tflite_model_quant = converter.convert()

    # Create temporary file to calculate size of quantized model
    _, tflite_file_to_zip = tempfile.mkstemp('.tflite')
    with open(tflite_file_to_zip, 'wb') as f:
        f.write(tflite_model_quant)
    wandb.log({"model_size [KB]": get_gzipped_model_size(tflite_file_to_zip) // 1000})

    # Save the quantized model
    tflite_model_file = pathlib.Path(main_folder) / "tf_model_quantized.tflite"
    tflite_model_file.write_bytes(tflite_model_quant)

    return tflite_model_quant
