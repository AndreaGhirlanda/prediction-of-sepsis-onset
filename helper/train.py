import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from helper import dataset, metrics, test


def get_criterion(class_unbalance, device, focal_loss):
    # Using BCE with digits to have better numerical stability[]
    if focal_loss:
        return WeightedFocalLoss(alpha=class_unbalance, gamma=2, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1/class_unbalance], device=device))


def get_optimizer(model, lr, weight_decay):
    # torch.optim.SGD(model.parameters(), lr=lr)#, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(lr_schedule, optimizer, n_batches, epoches, stop_lr, lr_epoch_div, lr_mult_factor):
    scheduler = None
    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_epoch_div, lr_mult_factor)
    elif lr_schedule == "const":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
    elif lr_schedule == "warmup":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, stop_lr, epochs=epoches, steps_per_epoch=n_batches)
    return scheduler


def train_nn_batch(wandb, dataloader, model, device, dtype, criterion, optimizer, scheduler, pbar, debug_grad):
    train_loss, correct = 0, 0
    total = 0
    model.train()
    for batch, (data, target, ids) in enumerate(dataloader):
        data = data.type(dtype).to(device)
        target = target.type(dtype).to(device)

        data = data.permute((0, 2, 1))

        # forward + backward + optimize
        output = model(data, ids)

        sigmoid = nn.Sigmoid()
        output_digit = sigmoid(output)

        loss = criterion(output, target)

        train_loss += loss.item()
        _, correct_batch, total_batch = metrics.accuracy(target, output_digit)
        correct += correct_batch
        total += total_batch

        # wandb.watch(model, log='all', log_freq=10)
        wandb.log({"train/loss": loss.item(),
                    "train/accuracy": correct/total,
                    "train/ones": output_digit.round().sum(),
                    "train/LR": optimizer.param_groups[0]['lr']})

        # zero the parameter gradients and compute backprop
        optimizer.zero_grad()
        loss.backward()

        if debug_grad:
            for name, param in model.named_parameters():
                if param.grad is not None and param.data is not None:
                    wandb.log({"grads/" + name + ".grad": wandb.Histogram(
                                    sequence = param.grad.flatten().cpu(),
                                    num_bins = 64)})
                    wandb.log({"weights/" + name + ".data": wandb.Histogram(
                                    sequence = param.data.flatten().cpu(),
                                    num_bins = 64)})
        
        optimizer.step() # Only for OneCycleLR

        pbar.set_description(f'Batch pbar. Loss: {train_loss/total:.5f}, accuracy: {correct/total:.3f}')
        pbar.update(1)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
    
    if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    pbar.reset()
    return (train_loss/total, correct/total)


def train_nn(wandb, file_path, model, train_generator, test_generator, device, dtype, criterion, optimizer, scheduler, epoches, n_batches, debug_grad, class_unbalance):
    pbar = tqdm(total=n_batches, ascii=True, dynamic_ncols=True)

    epoches_pbar = tqdm(range(epoches), ascii=True, dynamic_ncols=True)
    for _ in epoches_pbar:
        # Train model
        train_loss, accuracy = train_nn_batch(wandb, train_generator, model, device, dtype, criterion, optimizer, scheduler, pbar, debug_grad)
        # Run evaluation
        val_acc, val_loss, roc_auc_score, pr_auc_score, conf_mat, f1, w_f1 = test.test_nn(test_generator, model, criterion, file_path, device, dtype, wandb, class_unbalance)

        torch.save(model.state_dict(), os.path.join(file_path, 'model.pth'))
        epoches_pbar.set_description(f'Epochs pbar. Train loss: {train_loss:.5f}, accuracy: {accuracy:.3f}, Val loss: {train_loss:.5f}, accuracy: {accuracy:.3f},')
    pbar.close()
    epoches_pbar.close()


def train_forest(file_path, model, train_generator, test_generator, data_period, features, wandb, class_unbalance):
    # Must batch it all into one for training and evaluation.
    # Throws exception otherwise.
    # RAM usage ok, so not investigating further.
    features_mrmr, target = dataset.feature_extraction(train_generator, data_period, features)

    model.fit(features_mrmr, target)

    # Saving model
    pickle.dump(model, open(os.path.join(file_path, 'model.pkl'), 'wb'))

    val_acc, roc_auc_score, pr_auc_score, conf_mat, f1, weighted_f1 = test.test_forest(test_generator, model, data_period, features, file_path, wandb, class_unbalance)
    
    #Update metric, s
    wandb.log({"eval/accuracy": val_acc,
                "eval/ROC-AUC": roc_auc_score,
                "eval/PR-AUC": pr_auc_score,
                "eval/F1": f1})

    print(f"Accuracy {val_acc:.2f}, ROCAUC {roc_auc_score:.2f}, PRAUC {pr_auc_score:.2f}, F1: {f1:.2f}")

    return val_acc, roc_auc_score, pr_auc_score, f1, weighted_f1


def train_xgboost(file_path, model, train_set, test_set, wandb):
    # Must batch it all into one for training and evaluation.
    # Throws exception otherwise.
    # RAM usage ok, so not investigating further.

    eval_set = [train_set,test_set]
    eval_metrics = ["error","logloss", "auc"]

    model.fit(
        *train_set, 
        eval_metric=eval_metrics,
        eval_set=eval_set, verbose=False
    )

    # Saving model
    pickle.dump(model, open(os.path.join(file_path, 'model.pkl'), 'wb'))

    results = model.evals_result()
    results_sets = list(results.keys())
    train_curves = results[results_sets[0]]
    test_curves = results[results_sets[1]]

    train_curve = {}
    for metric in eval_metrics:
        train_curve[metric] = wandb.plot.line_series(
                                xs=range(len(train_curves[metric])),
                                ys=[train_curves[metric],test_curves[metric]],
                                keys=["Training","Testing"],
                                title=f"Training {metric}",
                                xname="Iteration"
                            )

    wandb.log(train_curve)

    val_acc, roc_auc_score, pr_auc_score, conf_mat, f1 = test.test_xgboost(test_set, model, file_path, wandb)

    #Update metric
    wandb.log({"eval/accuracy": val_acc,
                "eval/ROC-AUC": roc_auc_score,
                "eval/PR-AUC": pr_auc_score,
                "eval/F1": f1,
                "eval/weighted_F1": weighted_f1})

    print(f"Accuracy {val_acc:.2f}, ROCAUC {roc_auc_score:.2f}, PRAUC {pr_auc_score:.2f}, F1: {f1:.2f}")
    return val_acc, roc_auc_score, pr_auc_score, f1
  
#code adapted from https://theguywithblacktie.github.io/kernel/machine%20learning/pytorch/2021/05/20/cross-entropy-loss.html#focal-loss-code-implementation  
class WeightedFocalLoss(nn.Module):
    def __init__(self, device, alpha=None, gamma=2):
        super(WeightedFocalLoss, self).__init__()

        if alpha is not None:
            alpha = torch.tensor([alpha, 1-alpha], device=device).cuda()
        else:
            print('Alpha is not given. Exiting..')
            exit()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # computed BCE loss
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # converting type of targets to torch.long
        targets  = targets.type(torch.long)

        # gather data along first axis
        at       = self.alpha.repeat(targets.data.shape[0], 1).gather(1, targets.data)

        # Creating the probabilities for each class from the BCE loss
        pt       = torch.exp(-BCE_loss)

        # Focal Loss formula
        F_loss   = at*(1-pt)**self.gamma * BCE_loss

        return F_loss.mean()
