import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tsaug
import numpy as np
import torch
import tensorflow as tf

from helper.dataset import peak_remover
   
def get_dataset(folder: str, added_noise: bool, peak_removal: bool, reduce_channels: bool) -> tuple:
    """
    Gets the dataset from the specified folder.

    :param folder: Path to the folder containing the dataset
    :param added_noise: Flag indicating whether to add noise to the data
    :param peak_removal: Flag indicating whether to remove peaks from the data
    :param reduce_channels: Flag indicating whether to reduce channels in the data
    :return: Tuple containing train_data, test_data, train_ids, and test_ids
    """    
    print("Loading dataset...")
    #read train data (ids and targer (0 or 1)) from txt file
    train_ids = pd.read_csv(os.path.join(folder, "train.txt"))
    train_ids["patientid"] = train_ids["patientid"].astype(str)
    #read test data (ids and targer (0 or 1)) from txt file
    test_ids = pd.read_csv(os.path.join(folder, "test.txt"))
    test_ids["patientid"] = test_ids["patientid"].astype(str)
    #read train data from csv file
    train_data = [pd.read_csv(os.path.join(folder, "ID_" + id + ".csv")).drop(['Unnamed: 0', 'date_time'], axis=1) for id in train_ids["patientid"]]
    #read test data from csv file
    test_data  = [pd.read_csv(os.path.join(folder, "ID_" + id + ".csv")).drop(['Unnamed: 0', 'date_time'], axis=1) for id in test_ids["patientid"]]
    
    #read unbalanced dataset (used for scaling)
    unbalanced_dataset = [pd.read_csv(os.path.join(folder, "ID_" + id + ".csv")).drop(['Unnamed: 0', 'date_time'], axis=1) for id in train_ids["patientid"].unique()]

    # remove mbp and glucose channels in case reduce_channels is set to True
    if reduce_channels:
        train_data = [df.drop(['mbp', 'glucose'], axis=1) for df in train_data]
        test_data = [df.drop(['mbp', 'glucose'], axis=1) for df in test_data]
        unbalanced_dataset = [df.drop(['mbp', 'glucose'], axis=1) for df in unbalanced_dataset]
    
    # get column names (used to reconvert train and test data to pandas dataframe after scaling)
    cols = train_data[0].columns


    # remove peaks in case peak_removal is set to True
    if peak_removal:
        train_data = [peak_remover(df) for df in train_data] 
        test_data = [peak_remover(df) for df in test_data]
        unbalanced_dataset = [peak_remover(df) for df in unbalanced_dataset]

    ####### scale data so that each feature is in the range [0,1]. This is necessary for quantization since with quantization all the values will have to be in the fixed range [-256, 255] #######
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(pd.concat(unbalanced_dataset)) # pd.concat is used to concatenate all the dataframes in the list unbalanced_dataset into a single dataframe in order to fit the scaler
    train_data = [pd.DataFrame(min_max_scaler.transform(df), columns=cols) for df in train_data] # apply the scaler to each dataframe in the list train_data. The scaler converts the dataframe into a numpy array, so we have to convert it back to a dataframe
    test_data = [pd.DataFrame(min_max_scaler.transform(df), columns=cols) for df in test_data]  # apply the scaler to each dataframe in the list test_data. The scaler converts the dataframe into a numpy array, so we have to convert it back to a dataframe

    # add noise to the data if flag added_noise is set to True
    if added_noise:
        ts_aug = tsaug.AddNoise(scale=0.05)
        train_data = [df.apply(lambda col: ts_aug.augment(col.to_numpy()), axis=0) for df in train_data]


    #Reorder columns (Needed for sensor fusion)
    reordered_cols = ['respiratory_rate', 'spo2', 'sbp', 'dbp', 'heart_rate', 'core_body_temperature'] if reduce_channels else ['respiratory_rate', 'spo2', 'sbp', 'dbp', 'heart0_rate', 'core_body_temperature', 'mbp', 'glucose']
    train_data = [df.reindex(columns=reordered_cols) for df in train_data]
    test_data = [df.reindex(columns=reordered_cols) for df in test_data]

    return train_data, test_data, train_ids, test_ids


