import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Sampler
import os
import sys
import math
from tqdm import tqdm

from . import sofa

import tsaug

from datetime import datetime
from typing import Dict, List, Tuple

def connect(params_dic: Dict) -> psycopg2.extensions.connection:
    """
    Establishes a connection to a PostgreSQL server using the provided parameters.

    :param params_dic: A dictionary containing connection parameters such as database name, user, password, host, and port.
    :return: A connection object representing the connection to the PostgreSQL server.
    """
    conn = None
    try:
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
        print("Connection error")
    return conn

def get_all_patient_ids(conn_params: Dict) -> pd.DataFrame:
    """
    Retrieves all patient IDs from a SQL database.

    :param conn_params: A dictionary containing connection parameters such as database name, user, password, host, and port.
    :return: A pandas DataFrame containing all patient IDs.
    """
    # Establish connection to the database
    conn = connect(conn_params)

    # Query all patient IDs from the database
    subject_ids = pd.read_sql_query("SELECT * FROM patientid_table", con=conn)

    # Close the database connection
    conn.close()

    return subject_ids

def in_sql_gen(data: list, begin: int = None, end: int = None) -> str:
    """
    Generates an SQL IN clause from a list of data.

    :param data: A list of values.
    :param begin: Optional. Start index of the slice of data to include.
    :param end: Optional. End index of the slice of data to include.
    :return: A string representing the generated SQL IN clause.
    """
    load = "("
    for i in data[begin:end]:
        load += str(i) + ", "
    load = load[:-2] + ")"
    return load


def sepsis_labeling(sofa: pd.DataFrame, pharma_table: pd.DataFrame, antibiotics_criteria: list) -> pd.Timestamp:
    """
    Labels sepsis onset time based on SOFA score and antibiotic administration.

    :param sofa: DataFrame containing SOFA scores over time.
    :param pharma_table: DataFrame containing pharmaceutical data including antibiotic administration times.
    :param antibiotics_criteria: List of criteria for identifying antibiotics administration in pharma_table.
    :return: Timestamp indicating the onset time of sepsis, or -1 if sepsis onset is not detected.
    """
    sofa_diff = sofa.iloc[:,-1] - sofa.iloc[:,-1].shift(1)
    onset = sofa_diff[sofa_diff >= 2].dropna().reset_index()
    if onset.shape[0] == 0:
        return -1
    else:
        # Return the first occurance where SOFA greater than treshold
        onset_index = onset["index"][0]
        sepsis_onset_time = sofa.loc[onset_index,"date_time"]

        suspition_infection = pharma_table[antibiotics_criteria].any(axis=1)
        suspition_infection_index = suspition_infection[suspition_infection == True].index

        if suspition_infection_index.shape[0] != 0:
            suspition_infection_index = suspition_infection_index[0] # Taking first antibiotic
            si_time = pharma_table.loc[suspition_infection_index, "givenat"]
            if (sepsis_onset_time > si_time - pd.Timedelta(hours=48)) and (sepsis_onset_time < si_time + pd.Timedelta(hours=24)):
                return sepsis_onset_time
            else:
                return -1
        else:
            return -1


def pad_hours_before(data: pd.DataFrame, freq: int, start_time: pd.Timestamp, zero_padding: bool) -> pd.DataFrame:
    """
    Pads missing data before the start time with zero values or backward fills using the last observed value.

    :param data: DataFrame containing the original data.
    :param freq: The frequency of data points in minutes.
    :param start_time: The start time of the data window.
    :param zero_padding: Whether to pad missing values with zeros.
    :return: DataFrame with missing data padded before the start time.
    """
    # Find the earliest timestamp in the data
    end = data["date_time"].min()
    
    # Generate timestamps to pad missing data
    time_pad = pd.date_range(start=start_time, end=end, freq=str(freq) + "Min", inclusive='neither')
    
    # Create DataFrame to pad missing data
    time_pad = pd.DataFrame([[t_p if c == "date_time" else data["patientid"][0] if c == "patientid" else np.NaN for c in data.columns] for t_p in time_pad.values], columns=data.columns)
    
    # Concatenate original data with padded data and sort by date_time
    data = pd.concat([data, time_pad], ignore_index=True).sort_values("date_time").reset_index(drop=True)
    
    # Fill missing values with zeros if zero_padding is True, otherwise backward fill
    if zero_padding:
        for col in data.columns:
            data[col] = data[col].replace('nan', np.nan).fillna(0)
    else:
        data = data.bfill()
    
    return data

def get_all_windows(data: pd.DataFrame, start_prediction_time: pd.Timestamp, prediction_time: int, online_training_interval: int, data_time: int, onset: pd.Timestamp, freq: str, start_from_beginning: bool, time_data: str, zero_padding: bool) -> List[pd.DataFrame]:
    """
    Generates all windows from the data.

    :param data: DataFrame containing the input data.
    :param start_prediction_time: Timestamp indicating the start time for making predictions.
    :param prediction_time: The time in minutes for onset prediction.
    :param online_training_interval: The interval between online trainings.
    :param data_time: The time in minutes given to train the network.
    :param onset: Timestamp indicating the onset time.
    :param freq: The period of data acquisition in minutes.
    :param start_from_beginning: Whether to use the full length of data available for both control and positive cases.
    :param time_data: Additional time data.
    :param zero_padding: Whether to use zero padding.
    :return: A list of DataFrames representing the generated windows.
    """
    windows = []
    if start_from_beginning:
        i = 1        
        window = get_window(data, prediction_time, data_time, onset, freq, shift = 0, start_from_beginning=start_from_beginning, time_data=time_data, zero_padding=zero_padding)
        while not window.empty:
            windows.append(window)
            window = get_window(data, prediction_time, data_time, onset, freq, shift = i*online_training_interval, start_from_beginning=start_from_beginning, online_training_interval=online_training_interval, time_data=time_data, zero_padding=zero_padding)
            i += 1
            
    else:
        nr_intervals = int((start_prediction_time - prediction_time) / online_training_interval) + 1

        for i in reversed(range(nr_intervals)):
            window = get_window(data, prediction_time + i*online_training_interval, data_time, onset, freq, time_data=time_data, start_from_beginning=start_from_beginning, online_training_interval=online_training_interval, zero_padding=zero_padding)
            windows.append(window)

    return windows 

def clean_window(window: pd.DataFrame, data_time: int, freq: str, col_interest: List[str]) -> pd.DataFrame:
    """
    Cleans and preprocesses the window data by calculating means for specific columns and resampling.

    :param window: DataFrame representing the window data.
    :param data_time: The time in minutes given to train the network.
    :param freq: The period of data acquisition in minutes.
    :param col_interest: List of columns of interest for preprocessing.
    :return: Cleaned and preprocessed DataFrame representing the window data.
    """
    # Calculate mean values for specific columns
    window["sbp"] = window[["invasive_systolic_arterial_pressure", "non_invasive_systolic_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["dbp"] = window[["invasive_diastolic_arterial_pressure", "non_invasive_diastolic_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["mbp"] = window[["invasive_mean_arterial_pressure", "non_invasive_mean_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["spo2"] = window[["peripheral_oxygen_saturation", "peripheral_oxygen_saturation_0"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["glucose"] = window[["glucose_molesvolume_in_serum_or_plasma", "glucose_molesvolume_in_serum_or_plasma_0", "glucose_molesvolume_in_serum_or_plasma_1"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)

    # Resample data
    window = window[col_interest]
    window = window.resample(str(freq)+"Min", on="date_time", closed='left').first().reset_index()
    
    # Making sure the time dimension is right
    window = window.iloc[0:int(data_time/freq),:]

    return window

def get_window(data: pd.DataFrame, prediction_minutes: int, data_minutes: int, onset: str, freq: str, time_data=None, shift: int = 0, start_from_beginning: bool = False, online_training_interval: int = -1, zero_padding: bool = False) -> pd.DataFrame:
    """
    Retrieves a window of data based on the prediction time, data time, onset of sepsis, and other parameters.

    :param data: DataFrame representing the entire dataset.
    :param prediction_minutes: Time in minutes for onset prediction.
    :param data_minutes: Time in minutes given to train the network.
    :param onset: Time of sepsis onset.
    :param freq: Period of data acquisition in minutes.
    :param time_data: Time to consider for training if starting from the beginning.
    :param shift: Shift in minutes for the start of the window.
    :param start_from_beginning: Indicates whether to start from the beginning of the data.
    :param online_training_interval: Interval between online trainings.
    :param zero_padding: Indicates whether to use zero padding.
    :return: DataFrame representing the window of data.
    """
    # Getting the data_minutes amount of data before prediction_minutes from sepsis onset
    start_data = data["date_time"].min()
    end_data = data["date_time"].max()
    ###
    offset_minutes = pd.Timedelta(minutes=0)
    start_window = 0
    end = 0
    data_timedelta = pd.Timedelta(minutes=data_minutes)
    prediction_timedelta = pd.Timedelta(minutes=prediction_minutes)
    data_prediction_timedelta = pd.Timedelta(minutes=data_minutes+prediction_minutes)
    
    # If we want to start from the beginning of the data
    if start_from_beginning:
        shift = pd.Timedelta(minutes=shift)
        if onset == '-1':
            if time_data == None:
                end = end_data
            else:
                if (end_data - start_data) < time_data:
                    end = end_data
                else:
                    end = start_data + time_data - prediction_timedelta  
        else:
            onset = pd.to_datetime(onset)
            available_data_timedelta = (onset - start_data) - data_prediction_timedelta
            shift_window = int(available_data_timedelta.seconds/60 + available_data_timedelta.days*24*60)
            offset_minutes = pd.Timedelta(minutes = shift_window%online_training_interval)
            end = onset - prediction_timedelta
        
        if onset == '-1' and time_data == None and (end_data - start_data) < data_timedelta:
            start_window = end_data - data_timedelta + shift
        elif onset == '-1' and time_data != None and (end - start_data) < data_prediction_timedelta:
            start_window = end - data_timedelta + shift
        elif onset != '-1' and (onset - start_data) < data_prediction_timedelta:
            start_window = onset - data_prediction_timedelta + shift
        else:
            start_window = start_data + offset_minutes + shift
            
        end_window = start_window + data_timedelta
        
        if end_window > end:
            df_empty = pd.DataFrame({'A' : []})
            return df_empty
        
        if start_window < start_data:
            data = pad_hours_before(data, freq, start_window, zero_padding)   
            
        temp = data[data["date_time"].between(start_window, end_window, inclusive='left')]
        
        return temp       
    
    else:
        

        # If no sepsis the first values from the patient are reported
        if onset == '-1':
            if time_data is None:
                if (end_data - start_data) >= data_prediction_timedelta:
                    # Data are long enough, we can take the first
                    onset = start_data + data_prediction_timedelta
                else:
                    # Data are not enough, will be padded before
                    onset = end_data
            else:
                if (end_data - start_data) >= time_data:
                    # Data are long enough, we can take the first
                    onset = start_data + time_data
                else:
                    # Data are not enough, will be padded before
                    onset = end_data

        else:
            onset = pd.to_datetime(onset)

        start = onset - data_prediction_timedelta
        end = onset - prediction_timedelta
        if start < start_data:
            # We have to pad hours before admission
            data = pad_hours_before(data, freq, start, zero_padding)
        temp = data[data["date_time"].between(start, end, inclusive='right')]
        
        return temp
    
    return temp



def inclusion_criteria(data, inter_columns, pharma, age, dur_stay, onset, time_from_onset, antibiotics, antibiotics_before):
    """
    Applies inclusion criteria to determine whether a patient should be included in the dataset.

    :param data: DataFrame containing patient data.
    :param inter_columns: Columns of interest for checking missing values.
    :param pharma: DataFrame containing pharmaceutical data.
    :param age: Age of the patient.
    :param dur_stay: Minimum duration of stay required for inclusion (in timedelta format).
    :param onset: Time of sepsis onset.
    :param time_from_onset: Minimum time from sepsis onset required for inclusion (in timedelta format).
    :param antibiotics: List of antibiotics given.
    :param antibiotics_before: Time before which antibiotics should be given (in timedelta format).
    :return: 1 if the patient should be excluded based on the criteria, otherwise 0.
    """
    # Duration of stay > 24h
    lenght_of_stay = data["date_time"].max() - data["date_time"].min()
    if lenght_of_stay < dur_stay:
        return 1
    
    # Sepsis onset
    if onset != -1:
        onset_from_admission = onset - data["date_time"].min()
        if onset_from_admission < time_from_onset:
            return 1

    # Check if there are missing values
    # if data[inter_columns].eq(0).any().any():
    #     return 1

    # Check whether antibiotic given before time
    pharma_first_n_huour = pharma[pharma["givenat"] < (pharma["givenat"].min() + antibiotics_before)]
    temp = pharma_first_n_huour[antibiotics]
    if pharma_first_n_huour[antibiotics].ne(0).any().any():
        return 1
    
    # Sample passed
    return 0


## COLLATE
def collate_dataset_gen(batch):
    """
    Collates a batch of data samples.

    :param batch: Batch of data samples.
    :return: The batch itself without any modification.
    """
    return batch


def collate_nn(batch):
    """
    Collates a batch of data samples for neural network training or evaluation.

    :param batch: Batch of data samples, each containing a window, an onset label, and an identifier.
    :return: A tuple containing the collated window tensor, the collated onset label tensor, and the collated identifier tensor.
    """
    # Stack the windows, onsets, and ids from the batch
    window = torch.stack([w for w, _, _, _ in batch])
    onset = torch.stack([o for _, o, _, _ in batch])
    ids = torch.stack([id for _, _, _, id in batch])
    
    # Return the collated batch as a tuple
    return (window, onset, ids)

def collate_nn_sequential(batch):
    """
    Collates a batch of neural network training or evaluation with sequential data.

    :param batch: Batch of sequential data samples, each containing multiple windows, onset labels, and identifiers.
    :return: A tuple containing the collated window tensor, the collated onset label tensor, and the collated identifier tensor.
    """
    window_list = []  # List to store windows from all samples
    onset_list = []   # List to store onset labels from all samples
    id_list = []      # List to store identifiers from all samples
    
    # Iterate over each sample in the batch
    for windows, onset, col, id in batch:
        # Iterate over each window in the sample
        for w in windows:
            window_list.append(w)   # Append window to window list
            onset_list.append(onset)  # Append onset label to onset list
            id_list.append(id)       # Append identifier to identifier list
    
    # Stack the windows, onsets, and ids from all samples
    window = torch.stack(window_list)
    onset = torch.stack(onset_list)
    ids = torch.stack(id_list)
    
    # Return the collated batch as a tuple
    return (window, onset, ids)

def peak_remover(df):
    """
    Remove peaks from a DataFrame by replacing values that exceed a certain threshold with the previous value.

    :param df: Input DataFrame containing potentially noisy data.
    :return: DataFrame with peaks removed.
    """
    df_replaced = df.copy()  # Create a copy of the input DataFrame
    
    # Create a mask where the ratio of each value to the previous value is greater than 1.5
    mask = (df_replaced / df_replaced.shift(1)) > 1.5
    
    # Replace values that exceed the threshold with the previous value
    df_replaced[mask] = df_replaced.shift(1)[mask]
    
    return df_replaced  # Return the DataFrame with peaks removed

def get_patients_fs(folder: str, train_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get patient IDs from a CSV file in the specified folder and filter them based on training IDs.

    :param folder: Folder containing the CSV file.
    :param train_ids: List of training IDs to filter patient IDs.
    :return: Filtered patient IDs dataframe and the original patient IDs dataframe.
    """
    # Read patient IDs from the CSV file
    subject_ids = pd.read_csv(os.path.join(folder, "0labels.txt"))
    
    # Convert patient ID to string type to ensure consistent comparison
    subject_ids["patientid"] = subject_ids["patientid"].astype(str)
    
    # Filter patient IDs based on training IDs
    ids = subject_ids[subject_ids["patientid"].isin(train_ids)]
    
    return ids, subject_ids  # Return filtered patient IDs dataframe and original patient IDs dataframe


def get_time_from_admission_to_sepsis(ids: pd.DataFrame, folder: str) -> np.array:
    """
    Calculate the time from admission to sepsis for each patient in the given DataFrame.

    :param ids: DataFrame containing patient IDs and sepsis information.
    :param folder: Folder containing the patient data files.
    :return: Array containing the time duration from admission to sepsis for each patient.
    """
    times_before_sepsis = []
    for _, patient in ids[ids["sepsis"]==True].iterrows():
        data = pd.read_csv(os.path.join(folder, str(patient["patientid"]) + ".csv"))
        time_admission = datetime.strptime(data["date_time"].min(), "%Y-%m-%d %H:%M:%S")
        time_sepsis = datetime.strptime(patient["sepsis_time"], "%Y-%m-%d %H:%M:%S")
        time_before_sepsis = time_sepsis - time_admission
        times_before_sepsis.append(time_before_sepsis)
    times_before_sepsis = np.array(times_before_sepsis)
    return times_before_sepsis
 
class Dataset(torch.utils.data.Dataset):
  # Torch data loader
  def __init__(self, patientid, conn_params, freq = 2):
    'Initialization'
    self.patientid = patientid
    self.conn_params = conn_params
    self.freq = freq

  def __len__(self):
    'Denotes the total number of samples'
    return self.patientid.shape[0]

  def __getitem__(self, index):
    'Generates one sample of data'
    conn = connect(self.conn_params)
    patientid = self.patientid.iloc[index]

    load = in_sql_gen(patientid)

    observation_table = pd.read_sql_query("SELECT * FROM combined_table WHERE patientid in"  + load, con=conn)

    # Pandas .mean() will detele all NaN columns
    # Adding back the columns with all NaN
    observation_table_columns = observation_table.columns
    observation_table = observation_table.resample(str(self.freq)+"Min", on="date_time").agg(np.nanmean).reset_index()
    observation_table[[i for i in observation_table_columns if i not in observation_table.columns]] = np.NaN
    observation_table = observation_table.ffill().bfill().fillna(0)

    pharma_table = pd.read_sql_query("SELECT * FROM combined_pharma_table WHERE patientid in"  + load, con=conn)
    if(pharma_table.shape[0] > 0):
        pharma_table = pharma_table.fillna(0)
        # Resampling at 1 sample per hour and summing the total amount of given drug
        # NOTE: patientid wrong due to sum
        pharma_table = pharma_table.resample("1H", on="givenat").agg(np.nansum).reset_index()

    conn.close()

    observation_table_sofa = observation_table.resample("1H", on="date_time").mean().reset_index()
    sofa_table = sofa.get_sofa(observation_table_sofa, pharma_table)

    col_interest = ["date_time", "heart_rate", "invasive_systolic_arterial_pressure", "non_invasive_systolic_arterial_pressure", "invasive_diastolic_arterial_pressure", "non_invasive_diastolic_arterial_pressure", "invasive_mean_arterial_pressure", "non_invasive_mean_arterial_pressure", "peripheral_oxygen_saturation", "peripheral_oxygen_saturation_0", "respiratory_rate", "core_body_temperature", "glucose_molesvolume_in_serum_or_plasma", "glucose_molesvolume_in_serum_or_plasma_0", "glucose_molesvolume_in_serum_or_plasma_1", "oxygen_partial_pressure_in_arterial_blood", "platelets_volume_in_blood", "bilirubintotal_molesvolume_in_serum_or_plasma", "glasgow_coma_score_verbal_response_subscore", "glasgow_coma_score_motor_response_subscore", "glasgow_coma_score_eye_opening_subscore", "hourly_urine_volume", "platelets_volume_in_blood", "bilirubintotal_molesvolume_in_serum_or_plasma", "creatinine_molesvolume_in_blood"]

    antibiotics_criteria = ["penicillin_50_000_uml","clamoxyl_inj_lsg","clamoxyl_inj_lsg_2g","augmentin_tabl_625_mg","co_amoxi_tbl_625_mg","co_amoxi_tbl_1g","co_amoxi_12_g_inj_lsg","co_amoxi_22g_inf_lsg","augmentin_12_inj_lsg","augmentin_inj_22g","augmentin_22_inf_lsg","augmentin_ad_tbl_1_g","penicillin_g_1_mio","kefzol_inj_lsg","kefzol_stechamp_2g","cepimex","cefepime_2g_amp","cepimex_amp_1g","fortam_1_g_inj_lsg","fortam_2g_inj_lsg","fortam_stechamp_2g","rocephin_2g","rocephin_2_g_inf_lsg","rocephin_1_g_inf_lsg","zinacef_amp_15_g","zinat_tabl_500_mg","zinacef_inj_100_mgml","ciproxin_tbl_250_mg","ciproxin_tbl_500_mg","ciproxin_200_mg100ml","ciproxin_infusion_400_mg","klacid_tbl_500_mg","klacid_amp_500_mg","dalacin_c_600_phosphat_amp","dalacin_c_kps_300_mg","dalacin_c_phosphat_inj_lsg_300_mg","dalacin_phosphat_inj_lsg_600_mg","clindamycin_kps_300_mg","clindamycin_posphat_600","clindamycin_posphat_300","doxyclin_tbl_100_mg","vibravenös_inj_lsg_100_mg_5_ml","erythrocin_inf_lsg","floxapen_inj_lsg","floxapen_inj_lsg_2g","garamycin","sdd_gentamycinpolymyxin_kps","tienam_500_mg","tavanic_tbl_500_mg","tavanic_inf_lsg_500_mg_100_ml","meropenem_500_mg","meropenem_1g","meronem_1g","meronem_500_mg","flagyl_tbl_500_mg","metronidazole_tabl_200_mg","metronidazole_inf_500_mg100ml","avalox_filmtbl_400_mg","avalox_inf_lsg_400_mg","norfloxacin_filmtbl_400_mg","noroxin_tabl_400_mg","tazobac_inf_4g","tazobac_2_g_inf","piperacillin_tazobactam_225_inj_lsg","rifampicin_filmtbl_600_mg","rifampicin_inf_lsg","rimactan_inf_300_mg","rimactan_kps_300_mg","rimactan_kps_600_mg","colidimin_tbl_200_mg","xifaxan_tabl_550_mg","bactrim_amp_40080_mg_inf_lsg","bactrim_forte_lacktbl","bactrim_inf_lsg","obracin_80_mg","vancocin_oral_kps_250_mg","vancocin__amp_500_mg"]

    sepsis = sepsis_labeling(sofa_table, pharma_table, antibiotics_criteria)
    
    inclusion_criteria_param = {
        "data": observation_table,
        "inter_columns": col_interest,
        "pharma": pharma_table,
        "age": 18,
        "dur_stay": pd.Timedelta(24, unit='h'),
        "onset": sepsis,
        "time_from_onset": pd.Timedelta(20, unit='h'),
        "antibiotics": antibiotics_criteria,
        "antibiotics_before": pd.Timedelta(7, unit='h')
    }

    inclusion = inclusion_criteria(**inclusion_criteria_param)

    data = observation_table[col_interest]
    return (patientid.values[0], data, sepsis, inclusion)


# %%
class DatasetFromFS(torch.utils.data.Dataset):
  # Torch data loader
  def __init__(self, folder, train_ids, freq = 2, data_time = 24*60, prediction_time = 6*60, added_noise=False, normalise=False, peak_remover=False, zero_padding=False):
    'Initialization'
    self.folder = folder
    self.train_ids = train_ids
    self.freq = freq
    self.data_minutes = data_time
    self.prediction_minutes = prediction_time
    self.added_noise = added_noise
    self.normalise = normalise
    self.peak_remover = peak_remover
    self.zero_padding = zero_padding

    self.ids, _ = get_patients_fs(folder, train_ids)

  def __len__(self):
    'Denotes the total number of samples'
    return self.ids.shape[0]

  def __getitem__(self, input):
    'Generates one sample of data'
    # If args not none it containes the time to zero pad the negative samples
    if isinstance(input, np.ndarray):
        index = input[0]
        time_data = input[1]
    else:
        index = input
        time_data = None

    label_row = self.ids.iloc[index]
    data = pd.read_csv(os.path.join(self.folder, str(label_row["patientid"]) + ".csv"))
    data['date_time'] = pd.to_datetime(data['date_time'])

    sepsis = label_row["sepsis"]
    sepsis_time = label_row["sepsis_time"]

    # Dataset saved with freq=2
    window = get_window(data, self.prediction_minutes, self.data_minutes, sepsis_time, freq=2, time_data=time_data, zero_padding=self.zero_padding)
    
    col_interest = ["date_time", "heart_rate", "sbp", "dbp", "mbp", "respiratory_rate", "core_body_temperature", "spo2", "glucose"]
    
    window = clean_window(window, self.data_minutes, self.freq, col_interest)

    if self.peak_remover:
        window.iloc[:,1:] = peak_remover(window.iloc[:,1:])

    if self.normalise:
        scaler = MinMaxScaler()
        window.iloc[:,1:] = scaler.fit_transform(window.iloc[:,1:])

    if self.added_noise:
        ts_aug = tsaug.AddNoise(scale=0.05)
        window.iloc[:,1:] = window.iloc[:,1:].apply(lambda col: ts_aug.augment(col.to_numpy()), axis=0)

    window_tensor = torch.tensor(window.iloc[:,1:].values)
    onset_tensor = torch.tensor([1 if sepsis else 0])
    
    columns = window.columns.to_list()
    index = torch.tensor([index])
    return (window_tensor, onset_tensor, columns, index)


# This sampler provides the data indices already divided in batched
# Which will be fed to the dataloader
class CustomSampler(Sampler):
    def __init__(self, folder, train_ids, batch_size, sampling, onset_matching=False):
        self.batch_size = batch_size
        self.onset_matching = onset_matching

        self.ids, _ = get_patients_fs(folder, train_ids)

        # Get the indices of positive and negative samples
        self.ids = self.ids.reset_index()
        self.pos_ids = self.ids[self.ids["sepsis"]==True].index.to_numpy()
        self.neg_ids = self.ids[self.ids["sepsis"]==False].index.to_numpy()

        # Concatenating the lenght to the positives so that are shuffled accordingly
        if self.onset_matching:
            times_before_sepsis = get_time_from_admission_to_sepsis(self.ids, folder)
            self.pos_ids = np.concatenate((self.pos_ids.reshape(-1, 1), times_before_sepsis.reshape(-1, 1)), axis=1)

        if sampling == "under" or sampling == "undersampling":
            self.steps = int(2*np.floor(self.pos_ids.shape[0]/self.batch_size))
        elif sampling == "over" or sampling == "oversampling":
            self.pos_ids = np.repeat(self.pos_ids, len(self.neg_ids)//len(self.pos_ids), axis=0)
            self.steps = int(2*np.floor(self.pos_ids.shape[0]/self.batch_size))
        else:
            ValueError("Must select either \"under\" or \"over\"")


    def __iter__(self):
        # Yield the indices of each batch
        np.random.shuffle(self.pos_ids)
        np.random.shuffle(self.neg_ids)

        for s in range(self.steps):
            p_id = self.pos_ids[s*int(self.batch_size/2):(s+1)*int(self.batch_size/2)]
            n_id = self.neg_ids[s*int(self.batch_size/2):(s+1)*int(self.batch_size/2)]
            
            if self.onset_matching:
                # Concatenating the lenght of the positive cases to the negatives for onset matching
                n_id = np.concatenate((n_id.reshape(-1, 1), p_id[:,1].reshape(-1, 1)), axis=1)
            
            y = np.concatenate((p_id, n_id))
            np.random.shuffle(y)

            yield y

    def __len__(self):
        # Return the number of batches
        return self.steps


class DatasetFromFS_sequential(torch.utils.data.Dataset):
    # Torch data loader
    def __init__(self, folder, train_ids, freq, data_time, prediction_time, start_prediction_time, online_training_interval, start_from_beginning, added_noise=False, normalise=False, peak_remover=False, zero_padding=False):
        'Initialization'
        self.folder = folder
        self.train_ids = train_ids
        self.freq = freq
        self.data_time = data_time
        self.prediction_time = prediction_time
        self.start_prediction_time = start_prediction_time
        self.online_training_interval = online_training_interval
        self.start_from_beginning = start_from_beginning
        self.added_noise = added_noise
        self.normalise = normalise
        self.peak_remover = peak_remover
        self.zero_padding = zero_padding

        # Load the labels
        self.ids, _ = get_patients_fs(folder, train_ids)

    def __len__(self):
        'Denotes the total number of samples'
        return self.ids.shape[0]

    def __getitem__(self, input):
        'Generates one sample of data'
        # If args not none it containes the time to zero pad the negative samples
        if isinstance(input, np.ndarray):
            index = input[0]
            time_data = input[1]
        else:
            index = input
            time_data = None
            
        label_row = self.ids.iloc[index]
        data = pd.read_csv(os.path.join(self.folder, str(label_row["patientid"]) + ".csv"))
        data['date_time'] = pd.to_datetime(data['date_time'])

        sepsis = label_row["sepsis"]
        sepsis_time = label_row["sepsis_time"]

        windows = get_all_windows(data, self.start_prediction_time, self.prediction_time, self.online_training_interval, self.data_time, sepsis_time, self.freq, self.start_from_beginning, time_data, self.zero_padding)

        columns = windows[0].columns.to_list()

        col_interest = ["date_time", "heart_rate", "sbp", "dbp", "mbp", "respiratory_rate", "core_body_temperature", "spo2", "glucose"]
        
        windows_tensor = []
        for w in windows:
            window = clean_window(w, self.data_time, self.freq, col_interest)
            
            if self.peak_remover:
                window.iloc[:,1:] = peak_remover(window.iloc[:,1:])

            if self.normalise:
                scaler = MinMaxScaler()
                window.iloc[:,1:] = scaler.fit_transform(window.iloc[:,1:])

            if self.added_noise:
                ts_aug = tsaug.AddNoise(scale=0.05)
                window.iloc[:,1:] = window.iloc[:,1:].apply(lambda col: ts_aug.augment(col.to_numpy()), axis=0)
            windows_tensor.append(torch.tensor(window.iloc[:,1:].values))

        onset_tensor = torch.tensor([1 if sepsis else 0])
        index = torch.tensor([index])

        return (windows_tensor, onset_tensor, columns, index)


def generate_dataset(data_path: str):
    """
    Generates dataset from the database and saves it to CSV files.

    Args:
        data_path (str): Path to the directory where the dataset files will be saved.
    """
    
    conn_params = {
            "host"      : "",
            "database"  : "",
            "user"      : "",
            "password"  : ""
        }

    # Batch inside the dataloader since we're getting them from the database
    data_params = {'batch_size': 1,
        'shuffle': True,
        'num_workers': 8}
    
    # Getting the all the patient ids
    subject_ids = get_all_patient_ids(conn_params)

    dataset_hirid = Dataset(subject_ids, conn_params)
    pbar = tqdm(total=dataset_hirid.__len__(), ascii=True, dynamic_ncols=True)
    dataset_generator = torch.utils.data.DataLoader(dataset_hirid, collate_fn=collate_dataset_gen, **data_params)

    with open(os.path.join(data_path, "0labels.txt"), 'w') as f:
        f.write("patientid,sepsis,sepsis_time\n")
    for batch in dataset_generator:
        pbar.update(data_params["batch_size"])
        for patid, wind, ons, inc in batch:
            if inc == 0:
                wind.to_csv(os.path.join(data_path, str(patid)+".csv"))
                with open(os.path.join(data_path, "0labels.txt"), 'a') as f:
                    f.write(f"{patid},{ons != -1},{ons}\n")



def get_train_test_subject_ids(subject_ids: list, test_split: float, splits: int) -> tuple:
    """
    Split the subject ids into training and testing sets.

    Args:
        subject_ids (list): List of subject ids.
        test_split (float): Percentage of data to be used for testing.
        splits (int): Number of splits for the data.

    Returns:
        tuple: A tuple containing the training and testing subject ids.
    """
    
    # Augmented data have a _, must split them without train/test spillage
    # Checking the "original" ones
    orig_subj_id = [x for x in subject_ids if "_" not in x]
    split_dim = len(orig_subj_id)/splits

    test_subject_ids = orig_subj_id[int(split_dim*test_split):int(split_dim*(test_split+1))]
    train_subject_ids = [x for x in subject_ids if x.split("_")[0] not in test_subject_ids]

    return (train_subject_ids, test_subject_ids)


def get_dataloaders(data_path: str, data_params: dict, datagen_params: dict, collate_fn: callable, k: float, k_splits: int, sequential: bool = False, custom_sampler: str = "", onset_matching: bool = False, fake_unbalance: bool = False) -> tuple:
    """
    Get data loaders for training and testing.

    Args:
        data_path (str): Path to the data.
        data_params (dict): Parameters for creating the dataset.
        datagen_params (dict): Parameters for generating data.
        collate_fn (callable): Function to collate the data.
        k (float): Percentage of data to be used for testing.
        k_splits (int): Number of splits for the data.
        sequential (bool): Flag indicating whether to use sequential data loading.
        custom_sampler (str): Custom sampler for the data.
        onset_matching (bool): Flag indicating whether to use onset matching.
        fake_unbalance (bool): Flag indicating whether to use fake unbalance.

    Returns:
        tuple: A tuple containing the training generator, testing generator, class unbalance, and number of batches.
    """
    
    subject_ids = pd.read_csv(os.path.join(data_path, "0labels.txt"))
    subject_ids["patientid"] = subject_ids["patientid"].astype(str)
    train_subjectids, test_subjectids = get_train_test_subject_ids(subject_ids["patientid"].to_list(), k, k_splits)
    # Check for train test spillage
    [[(print("ERROR! TRAIN TEST SPILLAGE"), sys.exit(1)) for a in test_subjectids if a == b.split("_")[0]] for b in train_subjectids]

    if sequential:
        if custom_sampler != "":
            train_sampler = CustomSampler(data_path, train_subjectids, datagen_params["batch_size"], custom_sampler, onset_matching)
            train_set = DatasetFromFS_sequential(data_path, train_subjectids, **data_params)
            train_generator = torch.utils.data.DataLoader(train_set, num_workers=datagen_params['num_workers'], collate_fn=collate_fn, batch_sampler=train_sampler)
        else:
            train_set = DatasetFromFS_sequential(data_path, train_subjectids, **data_params)
            train_generator = torch.utils.data.DataLoader(train_set, **datagen_params, collate_fn=collate_fn)
        
        test_set = DatasetFromFS_sequential(data_path, test_subjectids, **data_params)
        test_generator = torch.utils.data.DataLoader(test_set, **datagen_params, collate_fn=collate_fn)
    else:
        if custom_sampler != "":
            train_sampler = CustomSampler(data_path, train_subjectids, datagen_params["batch_size"], custom_sampler, onset_matching)
            train_set = DatasetFromFS(data_path, train_subjectids, **data_params)
            train_generator = torch.utils.data.DataLoader(train_set, num_workers=datagen_params['num_workers'], collate_fn=collate_fn, batch_sampler=train_sampler)
        else:
            train_set = DatasetFromFS(data_path, train_subjectids, **data_params)
            train_generator = torch.utils.data.DataLoader(train_set, **datagen_params, collate_fn=collate_fn)
        
        # Testing always on the actual data
        test_set = DatasetFromFS(data_path, test_subjectids, **data_params)
        test_generator = torch.utils.data.DataLoader(test_set, **datagen_params, collate_fn=collate_fn)
    
    n_batches = len(train_generator)

    if fake_unbalance:
            class_unbalance = 1
    else:
        class_unbalance = subject_ids[subject_ids.loc[:,"sepsis"]==True].shape[0]/subject_ids[subject_ids.loc[:,"sepsis"]==False].shape[0]

    return (train_generator, test_generator, class_unbalance, n_batches)


def tensor_to_df(data: torch.Tensor, columns: List[str], T_min: int = 2) -> pd.DataFrame:
    """
    Convert a tensor to a DataFrame.

    Args:
        data (torch.Tensor): The tensor to convert.
        columns (List[str]): List of column names for the DataFrame.
        T_min (int, optional): Time in minutes. Defaults to 2.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    subjects = data.shape[0]
    time_samples = data.shape[1]
    data = data.reshape(-1, data.shape[-1])
    df = pd.DataFrame(data, columns=columns)
    
    df['id'] = np.repeat(np.arange(subjects), time_samples)

    date_ranges = [pd.date_range(start=0, periods=time_samples, freq=f'{T_min}min') for i in range(subjects)]
    date_ranges_series = pd.concat([pd.DataFrame(d) for d in date_ranges])

    df["time"] = date_ranges_series.reset_index(drop=True)

    return df
