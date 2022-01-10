import copy
import os
import sys
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, ".")
from src.helper import get_point_outliers, server_evaluation, tandem_precision_recall_curve, move_legend_below_graph

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rc('font', family='serif')

figsize = (9, 2.5)


def plot_schema_2d_partition_outlier(ax):
    center_a = (0.3, 0.3)
    center_b = (0.7, 0.7)
    diameter = 0.2
    circle_a = plt.Circle(center_a, diameter, alpha=0.5, color="blue")
    circle_b = plt.Circle(center_b, diameter, alpha=0.5, color="red")
    ax.add_artist(circle_a)
    ax.add_artist(circle_b)
    num_points = 4
    line_points_x = [center_a[0] + i / num_points * (center_b[0] - center_a[0]) for i in range(num_points)]
    line_points_y = [center_a[1] + i / num_points * (center_b[1] - center_a[1]) for i in range(num_points)]
    center_c = line_points_x[1], line_points_y[1]
    circle_c = plt.Circle(center_c, diameter, alpha=0.3, color="black", lw=0)
    line_points_x = line_points_x[:2]
    line_points_y = line_points_y[:2]
    ax.add_artist(circle_c)
    ax.plot(line_points_x, line_points_y, c="black", marker=".", lw=0.0)
    ax.arrow(center_a[0], center_a[1], center_b[0] - center_a[0], center_b[1] - center_a[1], color="black",
             lw=0.7, head_width=0.05, shape="full", zorder=3, length_includes_head=True)
    ax.set_ylabel("dim 2")
    ax.set_xlabel("dim 1")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Schema")
    ax.annotate(text="A", xy=(center_a[0] + 0.01, center_a[1] - 0.07))
    ax.annotate(text="B", xy=(center_b[0] + 0.02, center_b[1] - 0.07))
    ax.annotate(text=r"$c_i$", xy=(center_b[0] - 0.1, center_a[1]))


def plot_schema_2d(ax):
    center_a = (0.3, 0.3)
    center_b = (0.7, 0.7)
    diameter = 0.2
    circle_a = plt.Circle(center_a, diameter, alpha=0.3, color="black", lw=0)
    circle_b = plt.Circle(center_b, diameter, alpha=0.3, color="black", lw=0)
    ax.add_artist(circle_a)
    ax.add_artist(circle_b)
    num_points = 4
    line_points_x = [center_a[0] + i / num_points * (center_b[0] - center_a[0]) for i in range(num_points)]
    line_points_y = [center_a[1] + i / num_points * (center_b[1] - center_a[1]) for i in range(num_points)]
    ax.plot(line_points_x, line_points_y, c="black", marker=".", lw=0.0)
    ax.arrow(center_a[0], center_a[1], center_b[0] - center_a[0], center_b[1] - center_a[1], color="black",
             lw=0.7, head_width=0.05, shape="full", zorder=3, length_includes_head=True)
    ax.set_ylabel("dim 2")
    ax.set_xlabel("dim 1")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Schema")
    ax.annotate(text="A", xy=(center_a[0] + 0.01, center_a[1] - 0.07))
    ax.annotate(text="B", xy=(center_b[0] + 0.02, center_b[1] - 0.07))


def plot_outlier_scores(thresh: float = 96.0):
    nrows = 5
    ncols = 6
    fig_o, axes_o = plt.subplots(nrows, ncols, sharey="all")
    fig_f, axes_f = plt.subplots(nrows, ncols, sharey="all")
    filepath = os.path.join(os.getcwd(), "results", "result.csv")
    raw_data = pd.read_csv(filepath)
    raw_data = raw_data[raw_data["repetition"] == 0]
    grouped_data = raw_data.groupby(by="client")
    fig_o.suptitle("Ondevice")
    fig_f.suptitle("Federated")
    percentile = thresh
    for i, (key, client_data) in enumerate(grouped_data):
        lo, go = get_point_outliers(os_ondevice=client_data["os_ondevice"],
                                    os_federated=client_data["os_federated"],
                                    percentile=percentile)
        oso = client_data["os_ondevice"]
        x_lo = [i for i in range(len(lo)) if lo[i]]
        x_go = [i for i in range(len(lo)) if go[i]]
        axes_o.flatten()[i].scatter(range(len(oso)), oso, marker=".", alpha=0.2)
        axes_o.flatten()[i].scatter(x_lo, oso[lo], marker=".", alpha=0.5)
        axes_o.flatten()[i].scatter(x_go, oso[go], marker=".", alpha=0.5)
        axes_o.flatten()[i].axhline(np.percentile(oso, percentile), color="black", ls="dotted")
        osf = client_data["os_federated"]
        axes_f.flatten()[i].scatter(range(len(oso)), osf, marker=".", alpha=0.2)
        axes_f.flatten()[i].scatter(x_lo, osf[lo], marker=".", alpha=0.5)
        axes_f.flatten()[i].scatter(x_go, osf[go], marker=".", alpha=0.5)
        print(np.percentile(oso, percentile))
        print(np.percentile(osf, percentile))
        axes_f.flatten()[i].axhline(np.percentile(osf, percentile), color="black", ls="dotted")
    plt.show()


def plot_partition_outliers_over_shift_distance():
    sns.set_palette(sns.cubehelix_palette(2))
    filepath = os.path.join(os.getcwd(), "results", "results_po_1000.csv")
    raw_data = pd.read_csv(filepath)
    grouped_by_exp_index = raw_data.groupby(by="repetition")

    max_reps = np.max(raw_data["repetition"])
    results = []

    for rep, df in grouped_by_exp_index:
        grouped_by_client = df.groupby(by="client")
        os_federated = [
            data[1]["os_federated"].to_numpy() for data in grouped_by_client
        ]
        res = server_evaluation(os_federated)
        shift_amount = rep / (max_reps - 1)
        [results.append([rep, client, shift_amount,
                         res[0][client], res[1][client]]) for client in range(len(res[0]))]

    result_df = pd.DataFrame(results, columns=["rep", "client", "shift-amount", "os-star", "p-value"])
    result_normal = result_df[result_df["client"] != 0]
    result_anomaly = result_df[result_df["client"] == 0]
    task_ab_clients = np.logical_and(result_normal["client"] < 20, result_normal["client"] >= 10)
    normal_mean = result_normal.groupby("rep").mean().sort_values(by="shift-amount").rolling(window=1,
                                                                                             center=True).mean()
    task_a_mean = result_normal[result_normal["client"] < 10].groupby("rep").mean().sort_values(
        by="shift-amount").rolling(window=1, center=True).mean()
    task_ab_mean = result_normal[task_ab_clients].groupby("rep").mean().sort_values(by="shift-amount").rolling(window=1,
                                                                                                               center=True).mean()
    task_b_mean = result_normal[20 <= result_normal["client"]].groupby("rep").mean().sort_values(
        by="shift-amount").rolling(window=1, center=True).mean()
    anomaly_mean = result_anomaly.sort_values(by="shift-amount").rolling(window=1, center=True).median()

    fig, axes = plt.subplots(1, 2)
    x = anomaly_mean["shift-amount"]
    axes[1].scatter(x, task_a_mean["p-value"], label="A")
    axes[1].scatter(x, task_ab_mean["p-value"], label="AB")
    axes[1].scatter(x, task_b_mean["p-value"], label="B")
    axes[1].scatter(x, anomaly_mean["p-value"], label="PO")
    for ax_index, ax in enumerate(axes[1:]):
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["A", "inbetween", "B"])
    axes[1].set_yscale("log")
    axes[1].legend()
    plot_schema_2d_partition_outlier(axes[0])
    plt.show()


def create_result_table(thresh: float = 96):
    data = pd.read_csv(os.path.join("results", "result.csv"))
    result_dfs = []

    def scores(tp, fp, fn):
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * ((prec * rec) / (prec + rec))
        return [prec, rec, f1]

    for key, client_data in data.groupby(by="client"):
        for rep, df in client_data.groupby(by="repetition"):
            df["local"] = (df["os_ondevice"] > np.percentile(df["os_ondevice"], thresh)).astype(int)
            df["global"] = (df["os_federated"] > np.percentile(df["os_federated"], thresh)).astype(int)
            df["global"][df["global"] == 1] = 2
            lo, go = get_point_outliers(os_ondevice=df["os_ondevice"], os_federated=df["os_federated"],
                                              percentile=thresh, percentile_federated=thresh)
            df["tandem"] = 0
            df["tandem"][lo] = 1
            df["tandem"][go] = 2

            #metrics: precision, recall, f1, kappa
            ground_truth = df["labels"]
            result_local = df["local"]
            result_global = df["global"]
            result_tandem = df["tandem"]

            tp_local = np.sum(np.logical_and(ground_truth > 0, result_local == ground_truth))
            tp_global = np.sum(np.logical_and(ground_truth > 0, result_global == ground_truth))
            tp_tandem = np.sum(np.logical_and(ground_truth > 0, result_tandem == ground_truth))

            fp_local = np.sum(np.logical_and(ground_truth == 0, result_local > 0))
            fp_global = np.sum(np.logical_and(ground_truth == 0, result_global > 0))
            fp_tandem = np.sum(np.logical_and(ground_truth == 0, result_tandem > 0))

            fn_local = np.sum(np.logical_and(ground_truth > 0, result_local == 0))
            fn_global = np.sum(np.logical_and(ground_truth > 0, result_global == 0))
            fn_tandem = np.sum(np.logical_and(ground_truth > 0, result_tandem == 0))

            share_global = 4 * (rep % 6 / 5.0)
            share_local = 4 - share_global
            outlier_distribution = r"{}%, {}%".format(round(share_local, 1), round(share_global, 1))

            scores_local = ["local", outlier_distribution] + scores(tp_local, fp_local, fn_local)
            scores_global = ["federated", outlier_distribution] + scores(tp_global, fp_global, fn_global)
            scores_tandem = ["tandem", outlier_distribution] + scores(tp_tandem, fp_tandem, fn_tandem)

            res = pd.DataFrame([scores_local, scores_global, scores_tandem],
                               columns=["Method", "Outlier Distr.", "Precision", "Recall", "F1"])
            result_dfs.append(res)
    result_df = pd.concat(result_dfs, ignore_index=True).groupby(by=["Outlier Distr.", "Method"]).mean().round(2).reset_index()
    print(result_df.to_latex(index=False))


def plot_kde_partition_outliers():
    sns.set_palette(sns.cubehelix_palette(n_colors=10))
    data = pd.read_csv(os.path.join("results", "result.csv"))
    data["shift"] = data["repetition"] / 9.0
    ax = plt.gca()
    for key, d in data.groupby(by="shift"):
        os_federated = [
            c_data["os_federated"].to_numpy() for _, c_data in d.groupby(by="client")
        ]
        os_star, probabilities = server_evaluation(os_federated)
        sns.kdeplot(os_star[0], label="{}".format(round(key, 2)), ax=ax)
    plt.ylabel("Density")
    plt.xlabel("Outlier score")
    move_legend_below_graph(np.asarray([ax]), ncol=5, title="Shift [std]")
    plt.show()


def latex_table_partition_outliers():
    data = pd.read_csv(os.path.join("results", "result.csv"))
    exp = np.floor(data["repetition"] / 50 + 2)
    nobs = np.power(10, exp)
    deviation = (data["repetition"] % 5) / 4.0
    pattern_std = 0.1
    deviation_std = deviation * pattern_std
    data["shift"] = deviation_std
    data["$|DB|$"] = nobs
    print(np.unique(deviation))
    bs = [1.0, 10.0, 100.0, 1000, "sqrt"]
    table_data = []
    data.sort_values(by=["$|DB|$"], inplace=True)
    for DB, db_data in data.groupby("$|DB|$"):
        for b in bs:
            row = [DB, b if b != "sqrt" else r"$\sqrt{{|DB|}}$"]
            for key, d in db_data.groupby(by="shift"):
                os_federated = [
                    c_data["os_federated"].to_numpy() for _, c_data in d.groupby(by="client")
                ]
                res = server_evaluation(os_federated, b)[1][0]
                row.append(round(res, 2))
            table_data.append(row)

    columns = list(np.unique(deviation))
    columns.insert(0, r"$|DB|$")
    columns.insert(1, "b")
    df = pd.DataFrame(table_data, columns=columns)
    print(df.to_latex(index=False, escape=False))


if __name__ == '__main__':
    latex_table_partition_outliers()