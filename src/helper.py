import os
from typing import List

import torch
import tsfel
from flwr.common.logger import log
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, t


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


def get_point_outliers(os_ondevice, os_federated, percentile=99,
                       is_outlier_confidence=0.99, classification_confidence=0.95):
    os_ondevice = np.array(os_ondevice)
    os_federated = np.array(os_federated)
    percentile_thresh_ondevice = np.percentile(os_ondevice, q=percentile)
    percentile_thresh_federated = np.percentile(os_federated, q=percentile)
    # the distance to the quantile
    diff_to_thresh_ondevice = np.array(os_ondevice) - percentile_thresh_ondevice
    diff_to_thresh_federated = np.array(os_federated) - percentile_thresh_federated
    # the confidence that the outlier score is on the correct side of the threshold
    confidence_ondevice = confidence(diff_to_thresh_ondevice)
    confidence_federated = confidence(diff_to_thresh_federated)
    all_outliers = np.logical_and(os_ondevice >= percentile_thresh_ondevice,
                                  confidence_ondevice > is_outlier_confidence)
    global_outlier_candidates = os_federated >= percentile_thresh_federated

    significant_global_outliers = np.logical_and(
        all_outliers,
        np.logical_and(
            global_outlier_candidates,
            confidence_federated > classification_confidence
        )
    )
    significant_local_outliers = np.logical_and(
        all_outliers,
        np.invert(significant_global_outliers)
    )

    log(logging.WARNING, "Number of local outliers: {}".format(np.sum(significant_local_outliers)))
    log(logging.WARNING, "Number of global outliers: {}".format(np.sum(significant_global_outliers)))

    return significant_local_outliers, significant_global_outliers


def server_evaluation(os_federated: np.ndarray):
    os_star = np.array([np.mean(os) for os in os_federated])
    os_star = (os_star - np.median(os_star)) / np.std(os_star)
    mean = np.mean(os_star)
    std = np.std(os_star, ddof=1)
    probs = [t.sf(val, loc=mean, scale=std, df=len(os_star)-1) for val in os_star]
    return np.array(os_star), np.array(probs)


def save_outlier_scores(client_indices: List[int], os_federated: List[np.ndarray], os_ondevice: List[np.ndarray],
                        labels: List[np.ndarray], exp_repetition):
    # create dataframe and store in "results"
    results_dir = os.path.join("results")
    file_path = os.path.join(results_dir, "result.csv")
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