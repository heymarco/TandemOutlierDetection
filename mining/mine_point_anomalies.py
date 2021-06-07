import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from src.helper import get_point_outliers, ipek_split_ratios


nrows = 5
ncols = 4


def __get_window__(data: pd.DataFrame, outlier_index: int, window_size: int = 40, stride:int = 20):
    start_index = outlier_index*stride
    end_index = start_index + window_size
    return data.iloc[start_index: end_index].index, data.iloc[start_index: end_index]


def plot_outlier_windows():
    lo, go, io = get_outlier_indices()

    fig, axes = plt.subplots(nrows, ncols)

    data_dir = os.path.join("data", "ipek")
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    filenames = os.listdir(data_dir)
    filenames = sorted(filenames)
    index = 0
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, sep=";")
        file.drop(["time"], axis=1, inplace=True)
        file = file[file["batC"] > 0.6]
        file.reset_index(drop=True, inplace=True)
        relevant_data = file[["accelX"]].apply(np.linalg.norm, axis=1)
        print(relevant_data)
        inliers = np.invert(
            np.logical_or(np.logical_or(lo[index], go[index]), io[index])
        )
        for i in range(len(inliers)):
            if inliers[i]:
                color = "black"
                alpha = 0.4
                lw = 0.3
                label = "inlier"
            elif lo[index][i]:
                color = "blue"
                alpha = 1.0
                lw = 0.8
                label = "local"
            elif go[index][i]:
                color = "red"
                alpha = 1.0
                lw = 0.8
                label = "global"
            elif io[index][i]:
                color = "orange"
                alpha = 1.0
                lw = 0.8
                label = "indecisive"
            x, y = __get_window__(relevant_data, outlier_index=i)
            axes.flatten()[index].plot(x, y, color=color, alpha=alpha, lw=lw, label=label)
        index += 1
    custom_lines = [Line2D([0], [0], color="black", lw=0.3, alpha=0.4),
                    Line2D([0], [0], color="blue", lw=0.8),
                    Line2D([0], [0], color="orange", lw=0.8),
                    Line2D([0], [0], color="red", lw=0.8)]
    plt.figlegend(custom_lines, ["inlier", "local", "indecisive", "global"],
                  loc='lower center', frameon=False, ncol=4)
    plt.show()




def plot_point_anomalies():
    filepath = os.path.join("results", "results.csv")
    raw_data = pd.read_csv(filepath)

    percentile = 99.7

    grouped_by_exp_index = raw_data.groupby(by="repetition")
    for tpl in grouped_by_exp_index:
        df = tpl[1]
        exp_results = []
        grouped_by_client = df.groupby(by="client")
        os_federated = [
            data[1]["os_federated"].to_numpy() for data in grouped_by_client
        ]
        os_ondevice = [
            data[1]["os_ondevice"].to_numpy() for data in grouped_by_client
        ]
        labels = [
            data[1]["labels"].to_numpy() for data in grouped_by_client
        ]

        fig_ondevice, axes_ondevice = plt.subplots(nrows, ncols)
        fig_federated, axes_federated = plt.subplots(nrows, ncols, sharey="all")
        fig_federated_gt, axes_federated_gt = plt.subplots(nrows, ncols)
        fig_ondevice_gt, axes_ondevice_gt = plt.subplots(nrows, ncols)

        for cid, (osf, oso, device_labels) in enumerate(zip(os_federated, os_ondevice, labels)):
            local_outliers, global_outliers, indecisive_outliers = get_point_outliers(os_ondevice=oso,
                                                                                      os_federated=osf,
                                                                                      percentile=percentile,
                                                                                      is_outlier_confidence=0.99,
                                                                                      classification_confidence=0.99)
            # plot inliers
            all_outliers = np.logical_or(
                local_outliers,
                np.logical_or(
                    global_outliers, indecisive_outliers
                )
            )

            inliers = np.invert(all_outliers)

            inlier_ground_truth = np.logical_or(
                device_labels == "benign",
                device_labels == "unlabeled"
            )

            outlier_ground_truth = np.invert(inlier_ground_truth)

            x = np.arange(len(oso))
            thresh = np.percentile(osf, percentile)
            axes_federated.flatten()[cid].scatter(x=x[inliers], y=osf[inliers],
                                                  label="Inliers", alpha=0.1)
            axes_federated.flatten()[cid].scatter(x=x[local_outliers], y=osf[local_outliers],
                                                  label="Local outliers", alpha=0.5)
            axes_federated.flatten()[cid].scatter(x=x[global_outliers], y=osf[global_outliers],
                                                  label="Global outliers", alpha=0.5)
            axes_federated.flatten()[cid].scatter(x=x[indecisive_outliers], y=osf[indecisive_outliers],
                                                  label="Indecisive", alpha=0.5)
            axes_federated.flatten()[cid].axhline(thresh, color="black")

            thresh = np.percentile(oso, percentile)
            axes_ondevice.flatten()[cid].scatter(x=x[inliers], y=oso[inliers],
                                                 label="Inliers", alpha=0.1)
            axes_ondevice.flatten()[cid].scatter(x=x[local_outliers], y=oso[local_outliers],
                                                 label="Local outliers", alpha=0.5)
            axes_ondevice.flatten()[cid].scatter(x=x[global_outliers], y=oso[global_outliers],
                                                 label="Global outliers", alpha=0.5)
            axes_ondevice.flatten()[cid].scatter(x=x[indecisive_outliers], y=oso[indecisive_outliers],
                                                 label="Indecisive", alpha=0.5)
            axes_ondevice.flatten()[cid].axhline(thresh, color="black")

            for label in np.unique(device_labels):
                if label == "benign" or label == "unlabeled":
                    axes_ondevice_gt.flatten()[cid].scatter(x=x[device_labels == label],
                                                            y=oso[device_labels == label],
                                                            label=label, alpha=0.1)
                    axes_federated_gt.flatten()[cid].scatter(x=x[device_labels == label],
                                                             y=osf[device_labels == label],
                                                             label=label, alpha=0.1)
                else:
                    print("Number of {} outliers: {}".format(label, np.sum(device_labels == label)))
                    axes_ondevice_gt.flatten()[cid].scatter(x=x[device_labels == label],
                                                            y=oso[device_labels == label],
                                                            label=label, alpha=0.3)
                    axes_federated_gt.flatten()[cid].scatter(x=x[device_labels == label],
                                                             y=osf[device_labels == label],
                                                             label=label, alpha=0.3)
                thresh = np.percentile(oso, percentile)
                axes_ondevice_gt.flatten()[cid].axhline(thresh, color="black")
                thresh = np.percentile(osf, percentile)
                axes_federated_gt.flatten()[cid].axhline(thresh, color="black")

        fig_ondevice.suptitle("On-device")
        fig_federated.suptitle("Federated")
        fig_ondevice_gt.suptitle("On-device gt")
        fig_federated_gt.suptitle("Federated gt")
        plt.legend()
        plt.show()


def get_outlier_indices():
    filepath = os.path.join("results", "results.csv")
    raw_data = pd.read_csv(filepath)

    percentile = 99.5

    grouped_by_exp_index = raw_data.groupby(by="repetition")
    for tpl in grouped_by_exp_index:
        df = tpl[1]
        grouped_by_client = df.groupby(by="client")
        os_federated = [
            data[1]["os_federated"].to_numpy() for data in grouped_by_client
        ]
        os_ondevice = [
            data[1]["os_ondevice"].to_numpy() for data in grouped_by_client
        ]
        labels = [
            data[1]["labels"].to_numpy() for data in grouped_by_client
        ]

        lo = []
        go = []
        io = []

        for cid, (osf, oso, device_labels) in enumerate(zip(os_federated, os_ondevice, labels)):
            local_outliers, global_outliers, indecisive_outliers = get_point_outliers(os_ondevice=oso,
                                                                                      os_federated=osf,
                                                                                      percentile=percentile,
                                                                                      is_outlier_confidence=0.99,
                                                                                      classification_confidence=0.99)
            lo.append(list(local_outliers))
            go.append((global_outliers))
            io.append(list(indecisive_outliers))

        return lo, go, io


def print_metrics():
    filepath = os.path.join("results", "results.csv")
    raw_data = pd.read_csv(filepath)

    percentile = 99.9

    grouped_by_exp_index = raw_data.groupby(by="repetition")
    results = []
    for tpl in grouped_by_exp_index:
        rep = tpl[0]
        df = tpl[1]
        exp_results = []
        grouped_by_client = df.groupby(by="client")
        os_federated = [
            data[1]["os_federated"].to_numpy() for data in grouped_by_client
        ]
        os_ondevice = [
            data[1]["os_ondevice"].to_numpy() for data in grouped_by_client
        ]
        labels = [
            data[1]["labels"].to_numpy() for data in grouped_by_client
        ]

        for cid, (osf, oso, device_labels) in enumerate(zip(os_federated, os_ondevice, labels)):
            local_outliers, global_outliers, indecisive_outliers = get_point_outliers(os_ondevice=oso,
                                                                                      os_federated=osf,
                                                                                      percentile=percentile,
                                                                                      is_outlier_confidence=0.99,
                                                                                      classification_confidence=0.99)

            inlier_ground_truth = np.logical_or(device_labels == "benign", device_labels == "unlabeled")
            outlier_ground_truth = np.invert(inlier_ground_truth)

            true_positive = np.sum(np.logical_and(outlier_ground_truth, global_outliers))
            false_positive = np.sum(np.logical_and(global_outliers, inlier_ground_truth))
            true_negative = np.sum(np.logical_and(inlier_ground_truth, np.invert(global_outliers)))
            false_negative = np.sum(np.logical_and(outlier_ground_truth, np.invert(global_outliers)))
            nin = np.sum(inlier_ground_truth)
            nout = np.sum(outlier_ground_truth)

            results.append([rep, cid,
                            true_positive, false_positive, true_negative, false_negative,
                            np.sum(inlier_ground_truth), np.sum(outlier_ground_truth)])


    results = pd.DataFrame(results, columns=["rep", "cid", "tp", "fp", "tn", "fn", "nin", "nout"])
    print(results)


if __name__ == '__main__':
    print_metrics()
    plot_point_anomalies()
