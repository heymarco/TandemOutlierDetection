import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, ".")
from src.helper import get_point_outliers, server_evaluation, move_legend_below_graph

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rc('font', family='serif')

figsize = (9, 2.5)


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


def latex_table_point_outliers(thresh: float = 96):
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
    exp = np.floor(5 - data["repetition"] / 50)
    nobs = np.power(10, exp).astype(int)
    deviation = round((data["repetition"] % 5) / 4.0, 2)
    data["shift"] = deviation
    data["$|DB|$"] = nobs
    bs = [1, 10, 100, 1000, 10000]
    table_data = []
    data.sort_values(by=["$|DB|$"], inplace=True)
    for DB, db_data in data.groupby("$|DB|$"):
        for b in bs:
            for shift, shift_data in db_data.groupby(by="shift"):
                for rep, rep_data in shift_data.groupby("repetition"):
                    row = [DB, b if b != "sqrt" else r"$\sqrt{{|DB|}}$", shift, rep]
                    os_federated = [
                        c_data["os_federated"].to_numpy() for _, c_data in rep_data.groupby(by="client")
                    ]
                    res = server_evaluation(os_federated, b)[1][0]
                    row.append(round(res, 2))
                    table_data.append(row)
    temp_df = pd.DataFrame(table_data, columns=["$|DB|$", "$b$", "shift", "rep", "$p$-value"])
    df = temp_df.pivot(columns="shift", index=["$|DB|$", "$b$", "rep"]).groupby(["$|DB|$", "$b$"]).mean().round(2).reset_index()
    print(df.to_latex(index=False, escape=False))


if __name__ == '__main__':
    latex_table_partition_outliers()