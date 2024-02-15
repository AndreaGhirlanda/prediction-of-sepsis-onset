import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from typing import Union

from helper import dataset, metrics, test




def get_criterion(class_unbalance: float, device: torch.device, focal_loss: bool) -> torch.nn.Module:
    """
    Retrieves the loss criterion for training the neural network model.

    :param class_unbalance: Class unbalance factor
    :param device: Device on which to run the model (e.g., CPU or GPU)
    :param focal_loss: Whether to use focal loss or not
    :return: Loss criterion module (e.g., BCEWithLogitsLoss or WeightedFocalLoss)
    """
    if focal_loss:
        return WeightedFocalLoss(alpha=class_unbalance, gamma=2, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1/class_unbalance], device=device))


def get_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Retrieves the optimizer for training the neural network model.

    :param model: Neural network model to optimize
    :param lr: Learning rate for the optimizer
    :param weight_decay: Weight decay (L2 penalty) for the optimizer
    :return: Optimizer module (e.g., Adam or SGD)
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(lr_schedule: str, optimizer: torch.optim.Optimizer, n_batches: int, epoches: int, stop_lr: float, lr_epoch_div: int, lr_mult_factor: float) -> Union[torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.OneCycleLR, None]:
    """
    Retrieves the learning rate scheduler for adjusting the learning rate during training.

    :param lr_schedule: Learning rate scheduling strategy ("step", "const", or "warmup")
    :param optimizer: Optimizer module for which to adjust the learning rate
    :param n_batches: Total number of batches in each epoch
    :param epoches: Total number of epochs for training
    :param stop_lr: Final learning rate for "warmup" scheduler
    :param lr_epoch_div: Learning rate decay epoch division for "step" scheduler
    :param lr_mult_factor: Learning rate multiplier factor for "step" scheduler
    :return: Learning rate scheduler module (e.g., StepLR, OneCycleLR) or None if lr_schedule is invalid
    """
    scheduler = None
    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_epoch_div, lr_mult_factor)
    elif lr_schedule == "const":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
    elif lr_schedule == "warmup":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, stop_lr, epochs=epoches, steps_per_epoch=n_batches)
    return scheduler


def train_nn_batch(wandb, dataloader, model, device, dtype, criterion, optimizer, scheduler, pbar, debug_grad):
    """
    Trains the neural network model for one epoch (one batch at a time).

    :param wandb: Weights & Biases logger object
    :param dataloader: DataLoader for iterating over the training dataset
    :param model: Neural network model to train
    :param device: Device on which to run the model (e.g., CPU or GPU)
    :param dtype: Data type for the tensors (e.g., torch.float32)
    :param criterion: Loss criterion for the optimization process
    :param optimizer: Optimizer module for updating the model parameters
    :param scheduler: Learning rate scheduler for adjusting the learning rate during training
    :param pbar: Progress bar object for tracking batch progress
    :param debug_grad: Flag indicating whether to log gradient information for debugging
    :return: Tuple containing the average loss and accuracy for the epoch
    """
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


def train_nn(wandb, file_path: str, model: torch.nn.Module, train_generator: torch.utils.data.DataLoader, test_generator: torch.utils.data.DataLoader, device: torch.device, dtype: torch.dtype, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Union[torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.OneCycleLR], epoches: int, n_batches: int, debug_grad: bool, class_unbalance: float):
    """
    Trains the neural network model and evaluates its performance.

    :param wandb: Wandb object for logging metrics and visualizations
    :param file_path: Path to save the trained model
    :param model: Neural network model to train
    :param train_generator: DataLoader for training data
    :param test_generator: DataLoader for testing data
    :param device: Device on which to run the model (e.g., CPU or GPU)
    :param dtype: Data type for the model (e.g., torch.float32)
    :param criterion: Loss criterion for training the model
    :param optimizer: Optimizer for updating the model parameters
    :param scheduler: Learning rate scheduler for adjusting the learning rate during training
    :param epoches: Total number of epochs for training
    :param n_batches: Total number of batches in each epoch
    :param debug_grad: Whether to log gradients and weights for debugging purposes
    :param class_unbalance: Class unbalance factor
    """
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
