import os
from typing import List

import scipy.stats
import torch
import tsfel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as stats
from tqdm import tqdm


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_point_outliers(os_ondevice, os_federated, percentile=99, percentile_federated = False):
    if not percentile_federated:
        print("Use percentile_federated = 99")
        percentile_federated = percentile
    os_ondevice = np.array(os_ondevice)
    os_federated = np.array(os_federated)
    percentile_thresh_ondevice = np.percentile(os_ondevice, q=percentile)
    percentile_thresh_federated = np.percentile(os_federated, q=percentile_federated)
    # the distance to the quantile
    local_outliers = np.logical_and(os_ondevice > percentile_thresh_ondevice,
                                    os_federated <= percentile_thresh_federated)
    global_outliers = np.logical_and(os_ondevice > percentile_thresh_ondevice,
                                     os_federated > percentile_thresh_federated)

    return local_outliers, global_outliers


def server_evaluation(os_federated: np.ndarray, b = "sqrt"):
    means = []
    DB = len(np.concatenate(os_federated))
    if b == "max":
        b = np.min([len(os) for os in os_federated])
    if b == "sqrt":
        b = int(np.sqrt(DB))
    b = int(b)
    for i, arr in enumerate(os_federated):
        arr = np.sort(arr)
        db = len(arr)
        aggregation_size = int(db / np.ceil(db / b))
        # print(b, db, aggregation_size)
        dim1 = int(len(arr) / aggregation_size)
        dim2 = aggregation_size
        arr = arr[:dim1*dim2]
        means.append(np.reshape(arr, newshape=(dim1, dim2)).mean(axis=-1))

    p_values = []
    for i, device_mean in enumerate(means):
        dist_a = device_mean
        dist_b = []
        for j, m in enumerate(means):
            if j != i:
                dist_b += m.tolist()
        p = scipy.stats.mannwhitneyu(dist_a, dist_b, alternative="greater")
        p_values.append(p[1])

    return np.array(means), np.array(p_values)


def save_outlier_scores(client_indices: List[int], os_federated: List[np.ndarray], os_ondevice: List[np.ndarray],
                        labels: List[np.ndarray], exp_repetition):
    # create dataframe and store in "results"
    results_dir = os.path.join(os.getcwd(), "results")
    file_path = os.path.join(results_dir, "result.csv")
    print("Saving result under " + file_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    new_results = []
    for (client_index, scores_federated, scores_ondevice, labels_for_client) in zip(client_indices,
                                                                                    os_federated,
                                                                                    os_ondevice,
                                                                                    labels):
        for row in range(len(scores_federated)):
            osf = scores_federated[row]
            oso = scores_ondevice[row]
            label = labels_for_client[row]
            new_results.append([
                exp_repetition,
                client_index,
                osf,
                oso,
                label
            ])
    df = pd.DataFrame(new_results)
    if not os.path.exists(file_path):
        df.to_csv(file_path,
                  header=["repetition", "client", "os_federated", "os_ondevice", "labels"],
                  index=False)
    else:
        df.to_csv(file_path,
                  mode='a',
                  header=False,
                  index=False)


def extract_features_in_sliding_window(df: pd.DataFrame, window_size: int = 20, stride: int = 10) -> pd.DataFrame:
    cfg = tsfel.get_features_by_domain()
    df = df.iloc[:len(df) - (len(df) % window_size)]
    features = pd.concat([
        tsfel.time_series_features_extractor(cfg, df[i:i+window_size], verbose=0)
        for i in range(0, len(df), stride)
    ]).reset_index(drop=True)
    print("Length of features is {}".format(len(features)))
    return features


power_tool_split_ratios = [
    5600 / 14200,
    6800 / 10500,
    8700 / 19000,
    7000 / 15000,
    6500 / 18000,
    5500 / 11500,
    3900 / 10500,
    7200 / 12200,
    4400 / 8400,
    5050 / 11000,
    3600 / 7800,
    4900 / 11700,
    2800 / 8100,
    3500 / 7500,
    2400 / 6000
]


def centered_truncnorm(width_std, mean, std, size):
    my_mean = mean
    my_std = std
    myclip_a = my_mean - width_std * my_std
    myclip_b = my_mean + width_std * my_std
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def move_legend_below_graph(axes, ncol: int, title: str):
    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=ncol, title=title)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()