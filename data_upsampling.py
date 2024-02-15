# - Load data
# - Get class unbalance
# - For each positive create the same file times n (class imbalace)


import pandas as pd
import os
from tqdm import tqdm

data_path = "./hirid_oversample"
subject_ids = pd.read_csv(os.path.join(data_path, "0labels.txt"))

positive_subjects = subject_ids[subject_ids["sepsis"] == True]
expansion_rate = int((subject_ids.shape[0] - positive_subjects.shape[0]) / positive_subjects.shape[0])

with open(os.path.join(data_path, "0labels.txt"), 'a') as f:
    for index, sub in tqdm(positive_subjects.iterrows()):
        for i in range(expansion_rate-1):
            f.write((str(sub["patientid"]))+"_"+str(i)+","+str(sub["sepsis"])+","+str(sub["sepsis_time"])+"\n")
            data_csv = pd.read_csv(os.path.join(data_path, str(sub["patientid"])+".csv"))
            data_csv.to_csv(os.path.join(data_path, str(sub["patientid"])+"_"+str(i)+".csv"), index=False)
