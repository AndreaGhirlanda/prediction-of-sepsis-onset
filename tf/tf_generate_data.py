import torch
import numpy as np
import pandas as pd
import tqdm
import os
from helper import dataset
import warnings
import sys
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import tsaug
from typing import List, Tuple, Dict, Any


class DatasetFromFS(torch.utils.data.Dataset):
  # Torch data loader
  def __init__(self, folder, train_ids, freq, store_folder, prediction_time):
    'Initialization'
    self.folder = folder
    self.train_ids = train_ids
    self.freq = freq
    self.data_minutes = 240
    self.prediction_minutes = prediction_time
    self.zero_padding = False
    self.store_folder = store_folder
    self.ids, _ = dataset.get_patients_fs(folder, train_ids)

  def __len__(self):
    'Denotes the total number of samples'
    return self.ids.shape[0]

  def __getitem__(self, input):
    'Generates one sample of data'
    index = input

    label_row = self.ids.iloc[index]
    data = pd.read_csv(os.path.join(self.folder, str(label_row["patientid"]) + ".csv"))
    data['date_time'] = pd.to_datetime(data['date_time'])

    sepsis = label_row["sepsis"]
    sepsis_time = label_row["sepsis_time"]

    ####### Get the admission to sepsis time (Used only in admission-to-sepsis analysis) #######
    if sepsis_time == "-1":
       admission_to_sepsis_time = "NaN"
    else:
        admission_to_sepsis_time = pd.to_datetime(sepsis_time) - data["date_time"].min()


    # Dataset saved with freq=2 (if self.freq=60 the downsampling in granularity is done in clean_window where we take the mean over the 60 minutes)
    window = dataset.get_window(data, self.prediction_minutes, self.data_minutes, sepsis_time, freq=self.freq, time_data=None, zero_padding=self.zero_padding)
    
    # get the columns of interest (mbp and glucose are going to be removed in get_dataset (file: tf_get_data.py) if reduce_channels is set to True in the config file of wandb in tf_main.py)
    col_interest = ["date_time", "heart_rate", "sbp", "dbp", "mbp", "respiratory_rate", "core_body_temperature", "spo2", "glucose"]
    
    window = dataset.clean_window(window, self.data_minutes, self.freq, col_interest)
    window.iloc[:,1:] = dataset.peak_remover(window.iloc[:,1:])

    scaler = MinMaxScaler()
    window.iloc[:,1:] = scaler.fit_transform(window.iloc[:,1:])

    ts_aug = tsaug.AddNoise(scale=0.05)
    window.iloc[:,1:] = window.iloc[:,1:].apply(lambda col: ts_aug.augment(col.to_numpy()), axis=0)

    # Save the window in a csv file (The process of saving data for tf when looping through the train/test dataloader in save windows)
    # Note: distinction between train and test and the downsampling is done in the train.txt and test.txt files (see get_dataset in tf_get_data.py for more details)
    window.to_csv(os.path.join(self.store_folder ,f"ID_{label_row['patientid']}.csv"))
    with open(os.path.join(self.store_folder ,"train.txt"), 'a') as f:
        f.write(str(label_row["patientid"]) + "," + str(sepsis) + "," + str(admission_to_sepsis_time) + "\n")
        
    window_tensor = torch.tensor(window.iloc[:,1:].values)
    onset_tensor = torch.tensor([1 if sepsis else 0])
    
    columns = window.columns.to_list()
    index = torch.tensor([index])
    return (window_tensor, onset_tensor, columns, index)




def get_train_test_subject_ids(subject_ids: List[str], test_split: float, splits: int) -> Tuple[List[str], List[str]]:
    """
    Splits the list of subject IDs into training and testing sets.

    :param subject_ids: List of subject IDs
    :param test_split: Fraction of data to be used for testing
    :param splits: Number of splits for cross-validation
    :return: Tuple containing lists of training and testing subject IDs
    """
    # Augmented data have a _, must split them without train/test spillage
    # Checking the "original" ones
    orig_subj_id = [x for x in subject_ids if "_" not in x]
    split_dim = len(orig_subj_id)/splits

    test_subject_ids = orig_subj_id[int(split_dim*test_split):int(split_dim*(test_split+1))]
    train_subject_ids = [x for x in subject_ids if x.split("_")[0] not in test_subject_ids]

    return (train_subject_ids, test_subject_ids)

def get_dataloaders(data_path: str, datagen_params: Dict[str, Any], freq: int, store_folder: str, prediction_time: int, k: float, k_splits: int = 5, custom_sampler: str = "under") -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, float, int]:
    """
    Prepares data loaders for training and testing.

    :param data_path: Path to the data folder
    :param datagen_params: Parameters for data generation
    :param freq: Frequency of data sampling
    :param store_folder: Folder to store data
    :param prediction_time: Time for prediction
    :param k: Fraction of data to be used for testing
    :param k_splits: Number of splits for cross-validation
    :param custom_sampler: Type of custom sampler for training data
    :return: Tuple containing training and testing data loaders, class unbalance, and number of batches
    """
    subject_ids = pd.read_csv(os.path.join(data_path, "0labels.txt"))
    subject_ids["patientid"] = subject_ids["patientid"].astype(str)
    train_subjectids, test_subjectids = get_train_test_subject_ids(subject_ids["patientid"].to_list(), k, k_splits)
    # Check for train test spillage
    [[(print("ERROR! TRAIN TEST SPILLAGE"), sys.exit(1)) for a in test_subjectids if a == b.split("_")[0]] for b in train_subjectids]

    train_sampler = dataset.CustomSampler(data_path, train_subjectids, batch_size=32, sampling=custom_sampler)
    train_set = DatasetFromFS(data_path, train_subjectids, freq, store_folder, prediction_time)
    train_generator = torch.utils.data.DataLoader(train_set, num_workers=10, collate_fn=dataset.collate_nn, batch_sampler=train_sampler)
    
    # Testing always on the actual data
    test_set = DatasetFromFS(data_path, test_subjectids, freq, store_folder, prediction_time)
    test_generator = torch.utils.data.DataLoader(test_set, **datagen_params, collate_fn=dataset.collate_nn)
    

    ## n_batches and class_unbalance are not used
    n_batches = len(train_generator)
    class_unbalance = subject_ids[subject_ids.loc[:,"sepsis"]==True].shape[0]/subject_ids[subject_ids.loc[:,"sepsis"]==False].shape[0]

    return (train_generator, test_generator, class_unbalance, n_batches)


#this loop specifically saves the txt files associated to the windows. the csv files are also saved indirectly by calling the dataset class in the dataloader loop. The saving of the csv files is done in the dataset class.
def save_windows(train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, epochs: int, store_folder: str) -> None:
    """
    Saves windows associated with the txt files.

    :param train_loader: DataLoader for training data
    :param test_loader: DataLoader for testing data
    :param epochs: Number of epochs
    :param store_folder: Folder to store data
    """
    print("Saving windows...")
    
    # Prepare the txt file for test
    with open(os.path.join(store_folder, "train.txt"), 'a') as f: 
        f.write("patientid,sepsis,admission_sepsis_time\n") 

    # Generate test data
    for batch, (data, target, ids) in enumerate(test_loader):
        pass 
        
    # Rename the train.txt file to test.txt
    os.rename(os.path.join(store_folder, "train.txt"), os.path.join(store_folder, "test.txt"))
    
    # Prepare the txt file for train
    with open(os.path.join(store_folder, "train.txt"), 'a') as f: 
        f.write("patientid,sepsis,admission_sepsis_time\n") 

    # Iterate over the epochs
    for i in range(epochs):
        # Iterate over the batches
        for batch, (data, target, ids) in enumerate(train_loader):
            pass
             
    print("Done!")

def generate_data(k: int, freq: int, prediction_time: int, original_folder: str, store_folder: str) -> None:
    """
    Generates data for training and testing.

    :param k: Value of k
    :param freq: Frequency
    :param prediction_time: Prediction time
    :param original_folder: Path to the original folder
    :param store_folder: Path to the folder to store generated data
    """
    datagen_params = {'batch_size': 32,
        'shuffle': True,
        'num_workers': 10}
    

    if not os.path.exists(store_folder):
        os.makedirs(store_folder, exist_ok=True)

        train_generator, test_generator, _, _ = get_dataloaders(original_folder, datagen_params, k=k, freq=freq, store_folder=store_folder, prediction_time=prediction_time)
        
        # Save 20 epochs of data
        save_windows(train_generator, test_generator, epochs=20, store_folder=store_folder)
    else:
        print("Data with requested properties already generated. Skipping...")