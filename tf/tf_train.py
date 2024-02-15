import numpy as np
import pandas as pd

import tensorflow as tf

from tf.tf_get_data import get_dataset
import wandb
from wandb.keras import WandbMetricsLogger
from typing import List

def train_model(model: tf.keras.Model, train_data: List[pd.DataFrame], train_ids: pd.DataFrame, test_data: List[pd.DataFrame], test_ids: pd.DataFrame, wandb, epoches: int, start_lr: float, batch_size: int) -> tf.keras.Model:
    """
    Train a model using the provided training data and labels, and evaluate it on the provided test data and labels.

    Args:
    - model: The TensorFlow Keras model to be trained.
    - train_data: A list of pandas DataFrames containing the training data.
    - train_ids: A pandas DataFrame containing the training labels.
    - test_data: A list of pandas DataFrames containing the test data.
    - test_ids: A pandas DataFrame containing the test labels.
    - wandb: An instance of the Weights & Biases object for logging.
    - epoches: The number of epochs for training.
    - start_lr: The initial learning rate for the optimizer.
    - batch_size: The batch size for training.

    Returns:
    - The trained TensorFlow Keras model.
    """    
    print("Training model...")
    #convert dataframes into tensorflow tensors
    X_train = tf.convert_to_tensor(np.array([data.values for data in train_data]).astype("float32"))
    y_train = tf.convert_to_tensor(np.array(train_ids["sepsis"]).astype("float32"))
    X_test = tf.convert_to_tensor(np.array([data.values for data in test_data]).astype("float32"))
    y_test = tf.convert_to_tensor(np.array(test_ids["sepsis"]).astype("float32"))
    
    #create batches
    buffer_size = len(train_data)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=buffer_size, seed=wandb.config.seed).batch(batch_size, drop_remainder=True)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size, drop_remainder=True)

    #create loss function and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)  #from_logits=False because we are using sigmoid activation function in the last layer
    optimizer = tf.keras.optimizers.Adam(learning_rate=start_lr, weight_decay=None)

    #create list of values and boundaries for learning rate scheduler (the scheduler is a piecewise constant decay that divides the learning rate by 5 every 4 epochs)
    lr_values = [wandb.config.start_lr, wandb.config.start_lr * 0.2, wandb.config.start_lr * 0.04, wandb.config.start_lr * 0.008, wandb.config.start_lr * 0.0016, wandb.config.start_lr * 0.00032, wandb.config.start_lr * 0.000064, wandb.config.start_lr * 0.0000128, wandb.config.start_lr * 0.00000256, wandb.config.start_lr * 0.000000512]
    lr_boundaries = [3,7,11,15,19,23,27,31,35]

    #create learning rate scheduler
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries, values=lr_values)

    #compile the model (metrics used: accuracy, ROC-AUC, PR-AUC) (run_eagerly=False because when it is set to True, the model runs slower since it runs the code line by line instead of making use of computation graph. It should be avoided for deployment but it is useful for debugging)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'), tf.keras.metrics.AUC(curve='PR', name='PR-AUC')], run_eagerly=False)

    #train the model (steps_per_epoch is set to the number of batches present in each epoch. With the generate_data function I generate 20 epochs worth of downsampled data, regardless of how many epoches are selected so I divide the train data by 20 and then divide by the batch size to get the number of batches in each epoch. This mean that if epoches is set to 10 in the wandb config file, the model will use only 10 epoches worth of data)
    model.fit(train_ds, epochs=epoches, validation_data=test_ds, callbacks=[WandbMetricsLogger('batch'), tf.keras.callbacks.LearningRateScheduler(lr_scheduler)], steps_per_epoch=int((len(train_data)/20)/batch_size))

    return model 