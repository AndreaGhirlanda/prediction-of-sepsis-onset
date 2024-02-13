import torch
import numpy as np
import pandas as pd
import tqdm
import os
import tensorflow as tf
import warnings
import wandb
import random
import argparse

from tf_get_data import get_dataset
from models.tf_tcn import get_TCN
from tf_train import train_model
from tf_model_quantization import convert
from tf_test import test_float_model
from tf_quantized_model_evaluation import evaluate_quantized_model
from tf_generate_data import generate_data
import sys

warnings.filterwarnings("ignore")


### Set seed for reproducibility ###
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--gen_dataset", help="Generate the dataset", type=bool, default=False)

  #Initialize config for wandb
  config = {
      "original_folder": "hirid_8h",
      "gen_dataset": parser.parse_args().gen_dataset,
      "freq": 2,
      "k": 0,
      "prediction_time": 240,
      "k_splits": 7,
      "seed": 42,
      "added_noise": False,
      "peak_removal": True,
      "reduce_channels": True,
      "num_channels": [16, 16],
      "kernel_size": 7,
      "dense_layers": [32,32],
      "output_size": 1,
      "epoches": 11,
      "batch_size": 64,
      "start_lr": 0.001,
      "max_pool": [2, 2],
      "single_tcn": False,
      "sensor_fusion": True
  }
  
  # Set number of channels based on the reduce_channels flag
  config["vital_signs"] = 6 if config["reduce_channels"] else 8
  
  # Initialize wandb
  wandb.init(project="tf", entity="aghirlanda", config=config, allow_val_change=True)

  # Set seed
  set_seed(wandb.config.seed)

  # the main folder is the base folder for everything that is going to be saved while running the script
  main_folder = "hirid_tf_folder"
  # the tf_data_folder is the folder where the dataset for tf is saved (and loaded)
  tf_data_folder = f"hirid_tf_folder/hirid_tf_data_pred_time{wandb.config.prediction_time//60}h_freq={wandb.config.freq}_k={wandb.config.k}"
  
  # generate dataset fol all the combinations of k, freq and prediction time and exit (this is going to be used to generate the dataset to do the experiments necessary to make the plots with the different frequencies and prediction times)
  if wandb.config.gen_dataset:
    print("Generating dataset for all splits")
    for k in range(wandb.config.k_splits):
      for f in [2, 60]:
        for pred_time in [60, 120, 180, 240]:
          tf_data_folder = f"hirid_tf_folder/hirid_tf_data_pred_time{pred_time//60}h_freq={f}_k={k}"
          generate_data(k, f, pred_time, wandb.config.original_folder, tf_data_folder)
    sys.exit()

  # generate dataset for the specified properties in case it doesn't already exist
  if not os.path.exists(tf_data_folder):
    print("Dataset with specified properties doesn't exist. Generating the dataset")
    generate_data(wandb.config.k, wandb.config.freq, wandb.config.prediction_time, wandb.config.original_folder, tf_data_folder)

  # get model parameters from wandb config
  model_params = {
     "num_channels": wandb.config.num_channels,
      "kernel_size": wandb.config.kernel_size,
      "dense_layers": wandb.config.dense_layers,
      "output_size": wandb.config.output_size,
      "max_pool": wandb.config.max_pool,
      "single_tcn": wandb.config.single_tcn,
      "vital_signs": wandb.config.vital_signs,
      "sensor_fusion": wandb.config.sensor_fusion,
  }

  # get training parameters from wandb config
  train_params = {
    "epoches": wandb.config.epoches,
    "batch_size": wandb.config.batch_size,
    "start_lr": wandb.config.start_lr
  }


  #preprocess and load dataset
  train_data, test_data, train_ids, test_ids = get_dataset(tf_data_folder, added_noise=wandb.config.added_noise, peak_removal=wandb.config.peak_removal, reduce_channels=wandb.config.reduce_channels)
  
  #get model with specified parameters
  model = get_TCN(**model_params)
  
  #train model
  model = train_model(model, train_data, train_ids, test_data, test_ids, wandb=wandb, **train_params)


  #Evaluate float model
  test_float_model(model, test_data, test_ids)


  #Quantize model to int8
  quantized_model = convert(model, train_data, main_folder=main_folder)


  #Evaluate quantized model
  evaluate_quantized_model(quantized_model, test_data, test_ids)


    