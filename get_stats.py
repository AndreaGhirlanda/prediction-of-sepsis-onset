import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

def collate(batch):
  los = [l for l, _, _, _, _ in batch]
  sepsis_from_admission = [f for _, f, _, _, _ in batch]
  sepsis_to_discharge = [t for _, _, t, _, _ in batch]
  # columns = batch[3][0]
  # time_diffs = [t for _, _, _, _, t in batch]

  return (los, sepsis_from_admission, sepsis_to_discharge)#, columns, time_diffs)


# %%
class DatasetForStats(torch.utils.data.Dataset):
  # Torch data loader
  def __init__(self, folder):
    'Initialization'
    self.folder = folder
    self.subject_ids = pd.read_csv(os.path.join(folder, "0labels.txt"))
    self.subject_ids["patientid"] = self.subject_ids["patientid"].astype(str)

  def __len__(self):
    'Denotes the total number of samples'
    return self.subject_ids.shape[0]

  def __getitem__(self, index):
    label_row = self.subject_ids.iloc[index]
    data = pd.read_csv(os.path.join(self.folder, str(label_row["patientid"]) + ".csv"))
    
    col_interest = ["date_time", "heart_rate", "invasive_systolic_arterial_pressure", "non_invasive_systolic_arterial_pressure", "invasive_diastolic_arterial_pressure", "non_invasive_diastolic_arterial_pressure", "invasive_mean_arterial_pressure", "non_invasive_mean_arterial_pressure", "peripheral_oxygen_saturation", "peripheral_oxygen_saturation_0", "respiratory_rate", "core_body_temperature", "glucose_molesvolume_in_serum_or_plasma", "glucose_molesvolume_in_serum_or_plasma_0", "glucose_molesvolume_in_serum_or_plasma_1", "oxygen_partial_pressure_in_arterial_blood", "platelets_volume_in_blood", "bilirubintotal_molesvolume_in_serum_or_plasma", "glasgow_coma_score_verbal_response_subscore", "glasgow_coma_score_motor_response_subscore", "glasgow_coma_score_eye_opening_subscore", "hourly_urine_volume", "platelets_volume_in_blood", "bilirubintotal_molesvolume_in_serum_or_plasma", "creatinine_molesvolume_in_blood"]
    data = data[col_interest]

    data['date_time'] = pd.to_datetime(data['date_time'])

    sepsis = label_row["sepsis"]
    sepsis_time = label_row["sepsis_time"]

    # Computing lenght of stay
    los = data['date_time'].max() - data['date_time'].min()
    if sepsis:
      sepsis_time = datetime.strptime(sepsis_time, '%Y-%m-%d %H:%M:%S')
      sepsis_from_admission = sepsis_time - data['date_time'].min()
      sepsis_to_discharge =  data['date_time'].max() - sepsis_time
    else:
      sepsis_from_admission = -1
      sepsis_to_discharge = -1

    # Frequency of data
    time_diffs = []
    for col in data.columns:
      sample = data[col] - data[col].shift()
      sample_times = data.loc[sample[sample != 0].index, "date_time"]
      sample_diffs = sample_times - sample_times.shift()
      time_diffs.extend(sample_diffs.to_list())

    return (los, sepsis_from_admission, sepsis_to_discharge, data.columns.to_list(), time_diffs)

batch_size = 32
dataset_hirid = DatasetForStats("hirid")
pbar = tqdm(total=np.ceil(dataset_hirid.__len__()/batch_size), ascii=True, dynamic_ncols=True)
dataset_generator = torch.utils.data.DataLoader(dataset_hirid, num_workers=32, collate_fn=collate, batch_size=batch_size)

los = []
sepsis_from_admission = []
sepsis_to_discharge = []
columns = []
time_diffs = []

for l, s_f_a, s_to_d in dataset_generator:
  pbar.update(1)
  los.extend(l)
  sepsis_from_admission.extend(s_f_a)
  sepsis_to_discharge.extend(s_to_d)
  # columns.extend(col)
  # time_diffs.extend(t_d)

los = np.array(los)
los = los.astype('timedelta64[h]').astype(int)
sepsis_from_admission = np.array(sepsis_from_admission)
sepsis_from_admission = sepsis_from_admission[sepsis_from_admission != -1]
sepsis_from_admission = sepsis_from_admission.astype('timedelta64[h]').astype(int)
sepsis_to_discharge = np.array(sepsis_to_discharge)
sepsis_to_discharge = sepsis_to_discharge[sepsis_to_discharge != -1]
sepsis_to_discharge = sepsis_to_discharge.astype('timedelta64[h]').astype(int)

hist_sepsis_from_admission = np.histogram(sepsis_from_admission, 256)

fig, ax = plt.subplots(3, 1)
ax[0].hist(los, bins=256)
ax[0].set_title("Histogram lenght of stay")
ax[1].hist(sepsis_from_admission, bins=256)
ax[1].set_title("Time from admission to sepsis")
ax[2].hist(sepsis_to_discharge, bins=256)
ax[2].set_title("Time from sepsis to discharge")
plt.tight_layout()
plt.show()