import numpy as np
import pandas as pd

import tensorflow as tf
from helper import metrics
from tf_get_data import get_dataset
from tf_quantized_model_evaluation import get_confusion_matrix
import wandb



#NOTE: Uncomment the commented lines in order to save data for admission-to-sepsis-time analysis
def test_float_model(model, test_data, test_ids):
    # Get test data
    X_test = tf.convert_to_tensor(np.array([data.values for data in test_data]).astype("float32"))
    y_test = tf.convert_to_tensor(np.array(test_ids["sepsis"]).astype("float32"))

    # Initialize lists to store y_true and y_pred
    y_pred_list = []

    #final_df_list = [] # for admission-to-sepsis-time analysis


    # Iterate through test data and get y_true and y_pred
    for i in range(X_test.shape[0]):
        X = tf.reshape(X_test[i], (1, X_test[i].shape[0], X_test[i].shape[1]))
        y_pred = model(X)

        y_pred = 1 if y_pred.numpy()[0][0] >= 0.5 else 0

        y_pred_list.append(y_pred)



    ##### Calculate metrics #####

    # Get accuracy [%]
    accuracy = (np.sum(y_test == y_pred_list) * 100) / len(test_data)

    # Get AUROC and AUPRC
    _, _, auroc, _, _ = metrics.roc_curves(y_test, y_pred_list)
    _, _, auprc, _, _ = metrics.pr_curves(y_test, y_pred_list)

    # Get confusion matrix
    tp, fp, tn, fn = get_confusion_matrix(y_test, y_pred_list)

    # Log metrics to wandb
    wandb.log({"float/AUROC": auroc, "float/AUPRC": auprc, "float/accuracy": accuracy, "float/TP": tp, "float/FP": fp, "float/TN": tn, "float/FN": fn})