import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from helper import metrics

import matplotlib.pyplot as plt
import os
from typing import Any, Tuple

# Making the forward pass for testing 
# a generator because used in multple metrics
def get_predictions_nn(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    """
    Performs the forward pass of the neural network model on the test data generator.

    :param dataloader: DataLoader providing the test data
    :param model: Neural network model to evaluate
    :param device: Device on which to run the model (e.g., CPU or GPU)
    :param dtype: Data type to use (e.g., torch.float32)
    :return: None
    """
    model.eval()
    with torch.no_grad():
        for data, target, ids in tqdm(dataloader, desc='Test iterator', leave=False, dynamic_ncols=True):
            data = data.type(dtype).to(device)
            target = target.type(dtype).to(device)

            data = data.permute((0, 2, 1))
            output = model(data, ids)

            yield (target, output)


def test_nn(test_generator: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, file_path: str, device: torch.device, dtype: torch.dtype, wandb: Any, class_unbalance: float) -> Tuple[float, float, float, float, np.ndarray, float, float]:
    """
    Evaluates the performance of a neural network model on the test dataset.

    :param test_generator: DataLoader providing the test data
    :param model: Neural network model to evaluate
    :param criterion: Loss criterion used for evaluation
    :param file_path: File path for logging
    :param device: Device on which to run the model (e.g., CPU or GPU)
    :param dtype: Data type to use (e.g., torch.float32)
    :param wandb: Weights & Biases logging object
    :param class_unbalance: Class unbalance factor
    :return: Tuple containing evaluation metrics (accuracy, loss, ROC AUC, PR AUC, confusion matrix, F1 score, weighted F1 score)
    """
    total_loss = 0
    pred_list = []
    target_list = []
    for target, output in get_predictions_nn(test_generator, model, device, dtype):
        total_loss += criterion(output, target).item()

        sigmoid = nn.Sigmoid()
        output_digit = sigmoid(output)

        pred_list.append(output_digit)
        target_list.append(target)

    target_list = torch.cat(target_list).cpu().numpy()
    pred_list = torch.cat(pred_list).cpu().numpy()

    test_acc, _, total = metrics.accuracy(target_list, pred_list)

    total_loss = total_loss/total

    fpr, tpr, roc_auc, tresholds, roc_display = metrics.roc_curves(target_list[:,0], pred_list[:,0])
    precision, recall,pr_auc, thresholds,pr_display = metrics.pr_curves(target_list[:,0], pred_list[:,0])

    conf_mat = metrics.conf_matrix(target_list[:,0], pred_list[:,0])
    f1, weighted_f1 = metrics.f1(target_list[:,0], pred_list[:,0], class_unbalance)

    #######All the logs for the nn are here except for one in train_nn_batch#########

    ################LOG PLOTS################

    #PR Curve
    pr_display.plot(linewidth=2)
    wandb.log({"eval/PR-Curve": pr_display.figure_})

    #ROC Curve
    roc_display.plot(linewidth=2)
    wandb.log({"eval/ROC-Curve": roc_display.figure_})

    #Confusion Matrix
    wandb.log({"eval/conf_mat" : wandb.plot.confusion_matrix(preds=pred_list[:,0].round(), y_true=target_list[:,0])})

    ################LOG SCALAR VALUES################
    wandb.log({"eval/accuracy": test_acc,
                    "eval/loss": total_loss,
                    "eval/F1": f1,
                    "eval/weighted_F1": weighted_f1,
                    "eval/ROC-AUC": roc_auc,
                    "eval/PR-AUC": pr_auc})


    return (test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1, weighted_f1)

