# %%
from typing import Any
import torch
from torchsummary import summary
import os
import yaml
from yaml.loader import SafeLoader

import wandb

import random
import numpy as np

import pickle

from helper import dataset, train, test
from models import tcn

import warnings
warnings.filterwarnings('ignore')

import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm


# Used to color the output of terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


num_workers = int(os.cpu_count()-2)
print("CPU count: ", num_workers)

def get_model() -> tcn.TCN:
    """
    Generates a Temporal Convolutional Network (TCN) model with parameters specified in the config.

    :return: TCN model instance
    """
    net_params = {
        "data_minutes": int(wandb.config.data_time/wandb.config.data_freq_min),
        "output_size": 1,
        "vital_signs": 8,
        "num_channels": wandb.config.conv_filt,
        "dense_layers": wandb.config.dense_layers,
        "kernel_size": wandb.config.kernel_size,
        "dropout": wandb.config.dropout,
        "skip_conn": wandb.config.skip_conn,
        "batch_norm": wandb.config.batch_norm,
        "mid_dense": wandb.config.mid_dense,
        "out_mid_dense": wandb.config.out_mid_dense,
        "device": device,
        "dtype": dtype,
        "pos_enc": wandb.config.pos_enc,
        "single_tcn": wandb.config.single_tcn
    }
    return tcn.TCN(**net_params).to(device)


def train_net(epoches: int, lr_schedule: str, start_lr: float, stop_lr: float, lr_epoch_div: int, lr_mult_factor: float, weight_decay: float, online: bool, debug_grad: bool, custom_sampler: str, onset_matching: bool, fake_unbalance: bool, added_noise: bool, normalise: bool, peak_remover: bool, focal_loss: bool, zero_padding: bool, start_from_beginning: bool, sequential: bool) -> tuple[float, float, float, float, np.ndarray, float]:    
    """
    Trains the neural network.

    :param epoches: Number of epochs to train for
    :param lr_schedule: Learning rate scheduling method
    :param start_lr: Initial learning rate
    :param stop_lr: Final learning rate
    :param lr_epoch_div: Learning rate decay factor
    :param lr_mult_factor: Learning rate multiplication factor
    :param weight_decay: Weight decay factor
    :param online: Whether online training is enabled
    :param debug_grad: Whether to debug gradients
    :param custom_sampler: Custom data sampler
    :param onset_matching: Whether to match onsets
    :param fake_unbalance: Whether to apply fake unbalance
    :param added_noise: Whether added noise is enabled
    :param normalise: Whether data normalization is enabled
    :param peak_remover: Whether to remove peaks
    :param focal_loss: Whether to use focal loss
    :param zero_padding: Whether zero padding is enabled
    :param start_from_beginning: Whether to start training from the beginning
    :param sequential: Whether sequential training is enabled
    :return: Test accuracy, total loss, ROC AUC, PR AUC, confusion matrix, F1 score
    """
    if zero_padding:
        print("zero padding")
    data_params = {'data_time': wandb.config.data_time,
        'prediction_time': wandb.config.prediction_time,
        'freq': wandb.config.data_freq_min,
        'added_noise': added_noise,
        'normalise': normalise,
        'peak_remover': peak_remover,
        'zero_padding': zero_padding}

    datagen_params = {'batch_size': wandb.config.batch_size,
        'shuffle': True,
        'num_workers': num_workers}

    model = get_model()
    train_generator, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn, k=wandb.config.k, k_splits=wandb.config.k_splits, custom_sampler=custom_sampler, onset_matching=onset_matching, fake_unbalance=fake_unbalance)

    criterion = train.get_criterion(class_unbalance, device, focal_loss)
    optimizer = train.get_optimizer(model, start_lr, weight_decay)
    scheduler = train.get_scheduler(lr_schedule, optimizer, n_batches, epoches, stop_lr, lr_epoch_div, lr_mult_factor)

    train_params = {
        "wandb": wandb,
        "file_path": file_path,
        "model": model,
        "train_generator": train_generator,
        "test_generator": test_generator,
        "device": device,
        "dtype": dtype,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoches": epoches,
        "n_batches": n_batches,
        "debug_grad": debug_grad,
        "class_unbalance": class_unbalance,
    }

    if online:
        if sequential:
            print("Sequential online training")
            if start_from_beginning:
                print("start from beginning")
            time_steps = int((wandb.config.start_prediction_time - wandb.config.prediction_time)/wandb.config.online_training_interval) + 1
            if not start_from_beginning:
                datagen_params['batch_size'] = int(wandb.config.batch_size/time_steps) + 1
            data_params = {
                'data_time': wandb.config.data_time,
                'prediction_time': wandb.config.prediction_time,
                'freq': wandb.config.data_freq_min,
                'start_prediction_time': wandb.config.start_prediction_time,
                'online_training_interval': wandb.config.online_training_interval,
                'start_from_beginning': start_from_beginning,
                'zero_padding': zero_padding
            }
            train_generator, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn_sequential, k=wandb.config.k, k_splits=wandb.config.k_splits, custom_sampler=custom_sampler, sequential=sequential, onset_matching=onset_matching, fake_unbalance=fake_unbalance)

            criterion = train.get_criterion(class_unbalance, device, focal_loss)
            optimizer = train.get_optimizer(model, start_lr, weight_decay)
            scheduler = train.get_scheduler(lr_schedule, optimizer, n_batches, epoches, stop_lr, lr_epoch_div, lr_mult_factor)

            train_params["train_generator"] = train_generator
            train_params["test_generator"] = test_generator
            train_params["optimizer"] = optimizer
            train_params["scheduler"] = scheduler
            train.train_nn(**train_params)
        else:
            time_steps = int((wandb.config.start_prediction_time - wandb.config.prediction_time)/wandb.config.online_training_interval) + 1
            epoches_pbar = tqdm(range(time_steps), ascii=True, dynamic_ncols=True)
            for i in epoches_pbar:
                pred_time = wandb.config.start_prediction_time - i*wandb.config.online_training_interval
                epoches_pbar.set_description(f"prediction time: {pred_time} minutes")
                data_params['prediction_time'] = pred_time
                train_generator, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn, k=wandb.config.k, k_splits=wandb.config.k_splits, custom_sampler=custom_sampler, onset_matching=onset_matching, fake_unbalance=fake_unbalance)

                # Have to reinitialise optimiser and scheduler for each window
                optimizer = train.get_optimizer(model, start_lr, weight_decay)
                scheduler = train.get_scheduler(lr_schedule, optimizer, n_batches, epoches, stop_lr, lr_epoch_div, lr_mult_factor)

                train_params["train_generator"] = train_generator
                train_params["test_generator"] = test_generator
                train_params["optimizer"] = optimizer
                train_params["scheduler"] = scheduler
                train.train_nn(**train_params)

    else:
        train.train_nn(**train_params)


def test_net(focal_loss: bool, zero_padding: bool, start_from_beginning: bool, sequential: bool, online: bool, file_path: str, device: str, wandb: Any) -> tuple[float, float, float, float, Any, float]:
    """
    Tests a neural network.

    :param focal_loss: Whether to use focal loss
    :param zero_padding: Whether to use zero padding
    :param start_from_beginning: Whether to start testing from the beginning
    :param sequential: Whether to perform sequential testing
    :param online: Whether to perform online testing
    :param file_path: File path
    :param device: Device to run the tests on
    :param wandb: Object for logging to Weights & Biases
    :return: Tuple containing test accuracy, total loss, ROC AUC score, PR AUC score, confusion matrix, and F1 score
    """
    data_params = {'data_time': wandb.config.data_time,
        'prediction_time': wandb.config.prediction_time,
        'freq': wandb.config.data_freq_min,
        'added_noise': False,
        'normalise': wandb.config.normalise,
        'peak_remover': wandb.config.peak_remover,
        'zero_padding': zero_padding}

    datagen_params = {'batch_size': wandb.config.batch_size,
        'shuffle': True,
        'num_workers': num_workers}

    model = get_model()


    _, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn, k=wandb.config.k, k_splits=wandb.config.k_splits)
    criterion = train.get_criterion(class_unbalance, device, focal_loss)
    model.load_state_dict(torch.load(os.path.join(file_path, 'model.pth')))

    if online:
        time_steps = int((wandb.config.start_prediction_time - wandb.config.prediction_time)/wandb.config.data_freq_min) + 1
        epoches_pbar = tqdm(range(time_steps), ascii=True, dynamic_ncols=True)
        for i in epoches_pbar:
            pred_time = wandb.config.start_prediction_time - i*wandb.config.data_freq_min
            epoches_pbar.set_description(f"prediction time: {pred_time} minutes")
            pred_time = wandb.config.start_prediction_time - i*wandb.config.data_freq_min
            data_params['prediction_time'] = pred_time
            _, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn, k=wandb.config.k, k_splits=wandb.config.k_splits)
            print(f"Testing model with prediction time: {pred_time} minutes")
            test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1 = test.test_nn(test_generator, model, criterion, file_path, device, dtype, wandb, class_unbalance)
        if sequential:
            print("Sequential online testing")
            time_steps = int((wandb.config.start_prediction_time - wandb.config.prediction_time)/wandb.config.online_training_interval) + 1
            datagen_params['batch_size'] = int(wandb.config.batch_size/time_steps) + 1
            data_params = {
                'data_time': wandb.config.data_time,
                'prediction_time': wandb.config.prediction_time,
                'freq': wandb.config.data_freq_min,
                'start_prediction_time': wandb.config.start_prediction_time,
                'online_training_interval': wandb.config.online_training_interval,
                'start_from_beginning': start_from_beginning,
                'zero_padding': zero_padding
            }
            _, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn_sequential, k=wandb.config.k, k_splits=wandb.config.k_splits, sequential=sequential)

            test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1 = test.test_nn(test_generator, model, criterion, file_path, device, dtype, wandb)
            print(f"test_acc: {test_acc}, total_loss: {total_loss}, roc_auc: {roc_auc}, pr_auc: {pr_auc}, f1: {f1}")
        else:
            time_steps = int((wandb.config.start_prediction_time - wandb.config.prediction_time)/wandb.config.data_freq_min) + 1
            epoches_pbar = tqdm(range(time_steps), ascii=True, dynamic_ncols=True)
            for i in epoches_pbar:
                pred_time = wandb.config.start_prediction_time - i*wandb.config.data_freq_min
                epoches_pbar.set_description(f"prediction time: {pred_time} minutes")
                pred_time = wandb.config.start_prediction_time - i*wandb.config.data_freq_min
                data_params['prediction_time'] = pred_time
                _, test_generator, class_unbalance, n_batches = dataset.get_dataloaders(data_path, data_params, datagen_params, dataset.collate_nn, k=wandb.config.k, k_splits=wandb.config.k_splits)
                print(f"Testing model with prediction time: {pred_time} minutes")
                test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1 = test.test_nn(test_generator, model, criterion, file_path, device, dtype, wandb)
                print(f"test_acc: {test_acc}, total_loss: {total_loss}, roc_auc: {roc_auc}, pr_auc: {pr_auc}, f1: {f1}")
    else:
        test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1 = test.test_nn(test_generator, model, criterion, file_path, device, dtype, wandb, class_unbalance)
        print(f"test_acc: {test_acc}, total_loss: {total_loss}, roc_auc: {roc_auc}, pr_auc: {pr_auc}, f1: {f1}")
        
    return test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1


def argparser_init(argParser: argparse.ArgumentParser) -> None:
    """
    Initialize argument parser with various options.

    :param argParser: ArgumentParser object
    """    
    argParser.add_argument("--gen_dataset", help="Generate the dataset", action='store_true')
    argParser.add_argument("--train", help="Train the neural network", action='store_true')
    argParser.add_argument("--test", help="Test the neural network", action='store_true')
    argParser.add_argument("--sequential", help="Run sequential online prediction (online has to be set to true in order for this to have any effect)",type=bool)
    argParser.add_argument("--start_from_beginning", help="Use full length of data available for both control and positive cases", type=bool)
    argParser.add_argument("--zero_padding", help="Use Zero Padding (if False the first available value will copied for all the missing values instead)", type=bool)
    argParser.add_argument("--focal_loss", help="Use Focal Loss instead of BCE", type=bool)
    # argParser.add_argument("--online", help="Run online prediction", action='store_true')
    
    argParser.add_argument("--run_id", help="Load the model of the wandb run specified", type=str)
    argParser.add_argument("--start_lr", help="Set the starting learning rate", type=float)
    argParser.add_argument("--stop_lr", help="Set the stopping learning rate", type=float)
    argParser.add_argument("--lr_schedule", help="Set the learning rate schedule", type=str)
    argParser.add_argument("--epochs", help="Set the numnber of epochs", type=int)
    argParser.add_argument("--batch_size", help="Set the batch size", type=int)
    argParser.add_argument("--conv_filt", help="Set the number of embeddings", type=int)
    argParser.add_argument("--head", help="Set the number of heads", type=int)
    argParser.add_argument("--N", help="Set the N parameter of the transformer", type=int)
    argParser.add_argument("--dataset", help="Select which dataset to use", type=str)
    argParser.add_argument("--prediction_time", help="Set the time in minutes for onset prediction", type=int)
    argParser.add_argument("--data_time", help="Set the time in minutes given to train the network", type=int)
    argParser.add_argument("--data_freq_min", help="Set the period of data acquisition in minutes", type=int)
    argParser.add_argument("--seed", help="Set the random seed", type=int)
    argParser.add_argument("--online_training_interval", help="Set the interval between online trainings", type=int)
    argParser.add_argument("--mid_dense", help="Enables the dense layers after each channel of the TCN", type=int)
    argParser.add_argument("--out_mid_dense", help="Set the output dimension of the dense layer after the TCN", type=int)
    argParser.add_argument("--pos_enc", help="Enable the position encoding", type=bool)
    argParser.add_argument("--debug_grad", help="Enable the position encoding", type=bool)
    argParser.add_argument("--custom_sampler", help="Over/Undersample the dataset", type=str)
    argParser.add_argument("--onset_matching", help="Use onset matching", type=bool)
    argParser.add_argument("--fake_unbalance", help="Pass the balanced population to the loss weight.", type=bool)
    argParser.add_argument("--added_noise", help="Enable added noise.", type=bool)
    argParser.add_argument("--normalise", help="Enable normalisation.", type=bool)
    argParser.add_argument("--peak_remover", help="Enable the peak remover.", type=bool)


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    :param seed: The seed value to set
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    # Setting up arg parse
    argParser = argparse.ArgumentParser()
    argparser_init(argParser)
    args = argParser.parse_args()

    # Read config and init Wandb
    config={}
    file_path = ""
    if args.test:
        file_path = os.path.join("./", "trained", f'run_{args.run_id}')
        with open(os.path.join(file_path, 'config.yaml')) as f:
            config = yaml.load(f, Loader=SafeLoader)
    else:
        with open('config.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
    wandb_project = config["wandb_project"]

    if config["wandb"] and args.train:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    for key, value in vars(args).items():
        if key not in ["gen_dataset", "train", "test", "run_id"]:
            if value is not None:
                config[key] = value


    # allow_val_change for sweeps
    wandb.init(project=wandb_project, entity="deases-prediction", config=config, allow_val_change=True)

    # Setting seed for reproducibility
    set_seed(wandb.config.seed)

    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #mps
    # device = "cpu"

    data_path = os.path.join(".", wandb.config.dataset)

    if not wandb.config.wandb:
        print(bcolors.WARNING + "WARNING: running wandb in offline mode" + bcolors.ENDC)

    # Start training and testing
    if args.gen_dataset:
        dataset.generate_dataset(data_path)
    if args.train:
        file_path = os.path.join("./", "trained", f'run_{wandb.run.id}')
        os.makedirs(file_path)
        # Saving config as yaml file for 
        with open(os.path.join(file_path, 'config.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        train_param = {
            "focal_loss": wandb.config.focal_loss,
            "zero_padding": wandb.config.zero_padding,
            "start_from_beginning": wandb.config.start_from_beginning,
            "sequential": wandb.config.sequential,
            "online": wandb.config.online,
            "epoches" : wandb.config.epochs,
            "lr_schedule" : wandb.config.lr_schedule,
            "start_lr" : wandb.config.start_lr,
            "stop_lr" : wandb.config.stop_lr,
            "lr_epoch_div" : wandb.config.lr_epoch_div,
            "lr_mult_factor" : wandb.config.lr_mult_factor,
            "weight_decay": wandb.config.weight_decay,
            "debug_grad": wandb.config.debug_grad,
            "custom_sampler": wandb.config.custom_sampler,
            "onset_matching": wandb.config.onset_matching,
            "fake_unbalance": wandb.config.fake_unbalance,
            'added_noise': wandb.config.added_noise,
            'normalise': wandb.config.normalise,
            'peak_remover': wandb.config.peak_remover,
        }
        train_net(**train_param)
    if args.test:
        test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1 = test_net(wandb.config.focal_loss, wandb.config.zero_padding, wandb.config.start_from_beginning, wandb.config.sequential, wandb.config.online, file_path, device, wandb)
