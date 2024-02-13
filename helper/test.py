import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from helper import metrics, dataset

import matplotlib.pyplot as plt
import os

# Making the forward pass for testing 
# a generator because used in multple metrics
def get_predictions_nn(dataloader, model, device, dtype):
    model.eval()
    with torch.no_grad():
        for data, target, ids in tqdm(dataloader, desc='Test iterator', leave=False, dynamic_ncols=True):
            data = data.type(dtype).to(device)
            target = target.type(dtype).to(device)

            data = data.permute((0, 2, 1))
            output = model(data, ids)

            yield (target, output)


def test_nn(test_generator, model, criterion, file_path, device, dtype, wandb, class_unbalance):
    total_loss = 0
    pred_list = []
    target_list = []
    # TODO: verify it's correct
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

def test_forest(test_generator, model, data_period, features, file_path, wandb, class_unbalance):
    features_mrmr, target = dataset.feature_extraction(test_generator, data_period, features)
    # [pred class 0, pred class 1]
    pred = model.predict_proba(features_mrmr)[:,1]

    test_acc, _, total = metrics.accuracy(target, pred)

    _, _, roc_auc, _, disp_roc = metrics.roc_curves(target, pred, wandb)
    _, _, pr_auc, _, disp_pr = metrics.pr_curves(target, pred, wandb)

    # disp_roc.savefig(os.path.join(file_path, "roc.png"))
    # disp_pr.savefig(os.path.join(file_path, "precision_recall.png"))

    conf_mat = metrics.conf_matrix(target, pred, wandb)
    f1, weighted_f1 = metrics.f1(target, pred, class_unbalance)

    print(f"Accuracy {test_acc:.2f}, ROCAUC {roc_auc:.2f}, PRAUC {pr_auc:.2f}, F1 {f1:.2f}")
    return (test_acc, roc_auc, pr_auc, conf_mat, f1, weighted_f1)

def test_xgboost(test_set, model, file_path, wandb):
    features_mrmr, target = (test_set[0], test_set[1])
    # [pred class 0, pred class 1]
    pred = model.predict_proba(features_mrmr)[:,1]

    test_acc, _, total = metrics.accuracy(target, pred)

    _, _, roc_auc, _, disp_roc = metrics.roc_curves(target, pred, wandb)
    _, _, pr_auc, _, disp_pr = metrics.pr_curves(target, pred, wandb)

    # disp_roc.savefig(os.path.join(file_path, "roc.png"))
    # disp_pr.savefig(os.path.join(file_path, "precision_recall.png"))

    conf_mat = metrics.conf_matrix(target, pred, wandb)
    f1, weighted_f1 = metrics.f1(target, pred, class_unbalance)

    print(f"Accuracy {test_acc:.2f}, ROCAUC {roc_auc:.2f}, PRAUC {pr_auc:.2f}, F1 {f1:.2f}")
    return (test_acc, roc_auc, pr_auc, conf_mat, f1, weighted_f1)
