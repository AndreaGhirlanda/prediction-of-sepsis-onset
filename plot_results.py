import os
import yaml
from yaml.loader import SafeLoader

import matplotlib.pyplot as plt

import numpy as np

plt.rcParams.update({'font.size': 22})

with open('./results/tcn.yaml') as f:
    result = yaml.load(f, Loader=SafeLoader)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

def plot_metric_vs_time(ax, time, val_mean, val_std, metric, sampl_time):
    val_mean = np.array(val_mean)
    val_std = np.array(val_std)
    ax.plot(time, val_mean, 'o-', label=sampl_time, linewidth=3, markersize=10)
    ax.fill_between(time, val_mean-val_std, val_mean+val_std, alpha=0.2)
    ax.set_xticks(time)
    ax.set_xlabel("Prediction time")
    ax.set_ylabel(metric)
    return ax

for i, (metric, metrics) in enumerate(result.items()):
    for sampl_time, pred_times in metrics.items():
        pred_t = []
        val_mean = []
        val_std = []
        for pred_time, values in pred_times.items():
            pred_t.append(pred_time)
            values_arr = np.array(values)
            val_mean.append(values_arr.mean())
            val_std.append(values_arr.std())
            print(metric, sampl_time, pred_time, f"mean: {values_arr.mean():.3f}, std: {values_arr.std():.3f}")
        plot_metric_vs_time(ax[i], pred_t, val_mean, val_std, metric, sampl_time)
        ax[i].legend()

plt.tight_layout()
plt.show()

# ax[1].plot(x, tcn_60_min_roc, 'o-', label="60min", linewidth=3, markersize=10)
# ax[1].plot(x, tcn_2_min_roc, 'o-', label="2min", linewidth=3, markersize=10)
# ax[1].set_xticks(x)
# ax[1].set_xlabel("Prediction time")
# ax[1].set_ylabel("AUROC")
# ax[1].legend()

# plt.tight_layout()
# plt.savefig("results.pdf")

# # plt.show()