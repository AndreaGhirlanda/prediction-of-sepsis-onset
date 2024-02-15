import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
import os
from typing import List, Tuple


from tf.tf_get_data import get_dataset

from helper import metrics

import wandb

def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    """
    Calculates the elements of a confusion matrix.

    Args:
    - y_true: A list of true labels.
    - y_pred: A list of predicted labels.

    Returns:
    - A tuple containing the elements of the confusion matrix: (TP, FP, TN, FN)
    """
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    return tp, fp, tn, fn


def run_tflite_model(quantized_model: bytes, test_data: np.ndarray) -> List[int]:
    """
    Runs inference on a TFLite model.

    Args:
    - quantized_model: A bytes object containing the quantized TFLite model.
    - test_data: A numpy array containing the test data.

    Returns:
    - A list of predicted labels (0 or 1).
    """

    # Initialize the interpreter for the converted TFLite model
    interpreter = tf.lite.Interpreter(model_content=quantized_model)
    # Allocate memory for the input and output tensors
    interpreter.allocate_tensors()


    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((test_data.shape[0],), dtype=int)
    for i in range(test_data.shape[0]):
        test_input = test_data[i]

        # Rescale the input to int8
        input_scale, input_zero_point, _ = input_details["quantization_parameters"].values() # (https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter#get_input_details) Quantization parameters returns the parameters necessary to convert the input tensor from float32 to int8. It returns scale, zero point and quantized dimension
        test_input = (test_input / input_scale) + input_zero_point # equation to convert from float32 to int8  (https://www.tensorflow.org/lite/performance/quantization_spec)

        # adjust dimension of input tensor
        test_input = np.expand_dims(test_input, axis=0).astype(input_details["dtype"])
        # give input to interpreter
        interpreter.set_tensor(input_details["index"], test_input)
        # run inference
        interpreter.invoke()
        # get result of inference
        output = interpreter.get_tensor(output_details["index"])[0]
        # set y to 1 if output is greater than 0 (since we are using sigmoid activation function at the end of the modela and output is in range [-256, 255])
        predictions[i] = 1 if output[0] >= 0 else 0

    return predictions

def evaluate_quantized_model(quantized_model: bytes, test_data: List[pd.DataFrame], test_labels: pd.DataFrame) -> None:
    """
    Evaluate a quantized TFLite model on test data and log metrics on Weights & Biases.

    Args:
    - quantized_model: A bytes object containing the quantized TFLite model.
    - test_data: A list of pandas DataFrames containing the test data.
    - test_labels: A pandas DataFrame containing the test labels.

    Returns:
    - None
    """
    # Get Test dataset (y_test is already set to int8 since it can only be 0 or 1 while X_test will be converted to int8 later in run_tflite_model)
    X_test = tf.convert_to_tensor(np.array([data.values for data in test_data]).astype("float32"))
    y_test = np.array(test_labels["sepsis"]).astype("int8")

    #Get predictions
    predictions = run_tflite_model(quantized_model, X_test)

    # Get accuracy [%]
    accuracy = (np.sum(y_test == predictions) * 100) / len(test_data)

    # Get AUROC and AUPRC
    _, _, auroc, _, _ = metrics.roc_curves(y_test, predictions)
    _, _, auprc, _, _ = metrics.pr_curves(y_test, predictions)

    # Get confusion matrix
    tp, fp, tn, fn = get_confusion_matrix(y_test, predictions)

    # Log metrics on wandb
    wandb.log({"quant/AUROC": auroc, "quant/AUPRC": auprc, "quant/accuracy": accuracy, "quant/TP": tp, "quant/FP": fp, "quant/TN": tn, "quant/FN": fn})


    