import os
from typing import List

import scipy.stats
import torch
import tsfel
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ecdf(x):
    x = np.array(x, copy=True)
    assert len(x.shape) == 1
    x.sort()
    nobs = len(x)
    y = np.linspace(1.0 / nobs, 1, nobs)
    return x, y


def confidence(dist: np.ndarray):
    dist = np.abs(dist)
    nobs = len(dist)
    return 1 - np.exp(-2*nobs*dist)


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


def server_evaluation(os_federated: np.ndarray):
    means = []
    DB = len(np.concatenate(os_federated))
    b = int(np.sqrt(DB))
    for i, arr in enumerate(os_federated):
        arr = np.sort(arr)
        aggregation_size = b
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
    # df.to_csv(file_path,
    #           header=["repetition", "client", "os_federated", "os_ondevice", "labels"],
    #           index=False)
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


def tandem_precision_recall_curve(labels, probas_pred_local, probas_pred_federated, thresh_federated: float, pos_labels):
    percentiles = np.arange(len(labels) + 1) / len(labels) * 100
    precisions = [0.0]
    recalls = [1.0]
    for prob in tqdm(percentiles):
        local_outliers, global_outliers = get_point_outliers(probas_pred_local, probas_pred_federated,
                                                             percentile=prob, percentile_federated=thresh_federated)
        pred = np.full_like(local_outliers, 0, dtype=int)
        pred[local_outliers] = 1
        pred[global_outliers] = 2
        if isinstance(pos_labels, int):
            pred = pred == pos_labels
        else:
            try:
                pred = np.isin(pred, pos_labels)
            except:
                raise "Error: pos_labels must be integer or arr-like"

        TP = np.nansum(np.logical_and(pred, labels))
        FP = np.nansum(np.logical_and(pred, np.invert(labels)))
        FN = np.nansum(np.logical_and(np.invert(pred), labels))

        # Calculate true positive rate and false positive rate
        # Use try-except statements to avoid problem of dividing by 0
        try:
            precision = TP / (TP + FP)
        except:
            precision = 1

        try:
            recall = TP / (TP + FN)
        except:
            recall = 1

        precisions.append(precision)
        recalls.append(recall)
    precisions.append(1.0)
    recalls.append(0.0)
    return np.asarray(precisions), np.asarray(recalls)

ipek_split_ratios = [
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