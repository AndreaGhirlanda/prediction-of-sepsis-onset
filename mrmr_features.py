from helper import dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from mrmr import mrmr_classif

data_path = "./hirid_oversample"
data_params = {'data_time': 24,
                'prediction_time': 4,
                'freq': 2}

datagen_params = {'batch_size': 1024,
        'shuffle': True,
        'num_workers': 16}

if __name__ == '__main__':
    train_generator, _, _, _ = dataset.get_dataloaders(data_path, data_params, datagen_params, collate_fn=dataset.collate_forest, k=1, k_splits=5)
    data, target, columns = next(iter(train_generator))
    data_df = dataset.tensor_to_df(data, columns[1:], T_min=data_params["freq"])

    # parameter_extracted = MinimalFCParameters()
    parameter_extracted = EfficientFCParameters()
    features_filtered_direct = extract_features(timeseries_container=data_df, column_id='id', column_sort='time',
                                                default_fc_parameters=parameter_extracted)

    features_filtered_direct = features_filtered_direct.fillna(value=0)
    
    selected_features, scores, conf_matrix = mrmr_classif(X=features_filtered_direct, y=target, K=100, return_scores=True)

    print("Features name:")
    print(selected_features)
    print("------------------")

    selected_features_final = []
    scores_final = []
    for f, s in zip(selected_features, scores):
        # Check whether not added yet
        feature = f.split("__", maxsplit=1)[1]
        if feature not in selected_features_final:
            selected_features_final.append(feature)
            scores_final.append(s)
    fig, ax = plt.subplots(2, 1, figsize=(20,20))
    print("Best features:")
    for i, (f, s) in enumerate(zip(selected_features, scores)):
        print(f, s)
        ax[0].bar(i, s)
    ax[0].set_xticks(np.arange(0, len(selected_features), 1))
    ax[0].set_xticklabels(selected_features, rotation=45, ha='right')
    print("Features stripped:")
    for i, (f, s) in enumerate(zip(selected_features_final, scores_final)):
        print(f, s)
        ax[1].bar(i, s)
    ax[1].set_xticks(np.arange(0, len(selected_features_final), 1))
    ax[1].set_xticklabels(selected_features_final, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("mrmr.pdf")