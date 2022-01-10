import copy
import os
import sys
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, ".")
from src.helper import get_point_outliers, server_evaluation, tandem_precision_recall_curve

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


def plot_outlier_scores_over_distance(thresh: float = 99.0):
    sns.set_palette(["#000000"])
    filepath = os.path.join(os.getcwd(), "results", "result.csv")
    raw_data = pd.read_csv(filepath)
    raw_data["task"] = "A"
    raw_data["task"][np.logical_and(raw_data["client"] >= 10, raw_data["client"] < 20)] = "both"
    raw_data["task"][raw_data["client"] > 20] = "B"
    raw_data_cpy = copy.deepcopy((raw_data))
    raw_data["score"] = raw_data["os_ondevice"]
    raw_data["det."] = r"$L$"
    raw_data_cpy["score"] = raw_data_cpy["os_federated"]
    raw_data_cpy["det."] = r"$F$"
    raw_data = pd.concat((raw_data, raw_data_cpy))
    raw_data[r"$\delta$"] = raw_data["labels"]
    raw_data.sort_values(by="task", inplace=True)
    relevant_data = raw_data[raw_data["labels"] != -1]
    g = sns.FacetGrid(data=relevant_data, col="det.", row="task", sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x=r"$\delta$", y="score")

    plt.tight_layout()
    plt.show()


def plot_outlier_scores_over_distance_vertical(thresh: float = 99.0):
    sns.set_palette(["#000000"])
    fig = plt.figure(figsize=(3, 4))
    subfigs = fig.subfigures(nrows=3, ncols=1)
    filepath = os.path.join(os.getcwd(), "results", "result.csv")
    raw_data = pd.read_csv(filepath)

    def plot_data(subfig, data, title):
        axes = subfig.subplots(nrows=1, ncols=2, sharex=True, sharey=False)
        graph_data = data[data["labels"] != -1]
        sns.lineplot(data=graph_data, x="labels", y="os_ondevice", ci=None, ax=axes[0])
        sns.lineplot(data=graph_data, x="labels", y="os_federated", ci=None, ax=axes[1])
        subfig.suptitle(title)
        axes[0].set_ylabel(r"$os^L$")
        axes[1].set_ylabel(r"$os^F$")

    # first set of devices
    # plot_schema_2d(axes[0])
    relevant_data = raw_data[raw_data["client"] < 10]
    plot_data(subfigs[0], relevant_data, "Task A")
    # second set of devices
    relevant_data = raw_data[np.logical_and(raw_data["client"] >= 10, raw_data["client"] < 20)]
    plot_data(subfigs[1], relevant_data, "Task A and B")
    # third set of devices
    relevant_data = raw_data[raw_data["client"] >= 20]
    plot_data(subfigs[2], relevant_data, "Task B")
    plt.tight_layout()
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


def plot_evaluation_vary_nobs(filename: str = "synth_po_vary_nobs_10reps.csv"):
    # sns.set_palette(sns.color_palette("colorblind", n_colors=4))
    cached_path = os.path.join("results", "cache", filename)
    latex_columns = ["repetition", "client", r"$os^*_i$", r"$os_o$", "shift", r"$|db_i|$", "is PO", r"$p$-val"]
    if not os.path.exists(cached_path):
        print("Load file...")
        filepath = os.path.join(os.getcwd(), "results", filename)
        raw_data = pd.read_csv(filepath)
        print("File loaded; extracting data...")

        steps = 10
        nobs = []
        repetitions_with_same_params = 10
        for rep in raw_data["repetition"]:
            counter_for_params = int(rep / repetitions_with_same_params)
            nobs.append([100, 300, 1000, 3000][int(counter_for_params / steps)])
        is_outlier = []
        for client in raw_data["client"]:
            is_outlier.append(client == 0)
        raw_data["nobs"] = nobs
        raw_data["is po"] = is_outlier

        raw_data.sort_values(by=["repetition", "client"], inplace=True)
        grouped_by_repetition = raw_data.groupby(by="repetition")
        p_values = []
        for id, group in tqdm(grouped_by_repetition):
            os_federated = []
            number_of_observations = len(group[group["client"] == 0])
            for client, client_data in group.groupby(by="client"):
                os_federated.append(client_data["os_federated"])
                # hacky: we need to adjust labels for the plot! todo: fix it in code
                bool_array = np.logical_and(raw_data["repetition"] == id, raw_data["client"] == client)
                raw_data.loc[bool_array, "labels"] = np.array(group.loc[group["client"] == 0, "labels"])
            result = server_evaluation(os_federated)[1]
            result = np.expand_dims(result, 1).repeat(number_of_observations, 1)
            p_values += list(result.flatten())

        raw_data["p"] = p_values
        raw_data = raw_data.groupby(["repetition", "client"]).mean().reset_index()
        raw_data.columns = latex_columns
        raw_data.to_csv(cached_path, index=False)
    else:
        print("Using cache...")
        raw_data = pd.read_csv(cached_path)
    print(raw_data)
    raw_data.columns = latex_columns
    print("Extraction completed; plotting data...")
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    sns.lineplot(data=raw_data, x="shift", y=r"$os^*_i$", hue=r"$|db_i|$", ax=axes[0],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))
    sns.lineplot(data=raw_data, x="shift", y=r"$p$-val", hue=r"$|db_i|$", ax=axes[1],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))

    fig.set_size_inches(figsize)
    move_legend_below_graph(axes, ncol=8)
    axes[-1].set_yscale('log')
    plt.show()


def plot_evaluation_vary_affected_dims(filename: str = "synth_po_vary_dims_10reps.csv"):
    # sns.set_palette(sns.color_palette("colorblind", n_colors=4))
    cached_path = os.path.join("results", "cache", filename)
    latex_columns = ["repetition", "client", r"$os^*_i$", r"$os_o$", "shift", "dims", "is PO", r"$p$-val"]
    if not os.path.exists(cached_path):
        print("Load file...")
        filepath = os.path.join(os.getcwd(), "results", filename)
        raw_data = pd.read_csv(filepath)
        print("File loaded; extracting data...")

        steps = 10
        affected_dims = []
        repetitions_with_same_params = 10
        for rep in raw_data["repetition"]:
            counter_for_params = int(rep / repetitions_with_same_params)
            affected_dims.append([0.1, 0.3, 0.6, 1.0][int(counter_for_params / steps)])
        is_outlier = []
        for client in raw_data["client"]:
            is_outlier.append(client == 0)
        raw_data["dims"] = affected_dims
        raw_data["is po"] = is_outlier

        raw_data.sort_values(by=["repetition", "client"], inplace=True)
        grouped_by_repetition = raw_data.groupby(by="repetition")
        p_values = []
        for id, group in tqdm(grouped_by_repetition):
            os_federated = []
            number_of_observations = len(group[group["client"] == 0])
            for client, client_data in group.groupby(by="client"):
                os_federated.append(client_data["os_federated"])
                # hacky: we need to adjust labels for the plot! todo: fix it in code
                bool_array = np.logical_and(raw_data["repetition"] == id, raw_data["client"] == client)
                raw_data.loc[bool_array, "labels"] = np.array(group.loc[group["client"] == 0, "labels"])
            result = server_evaluation(os_federated)[1]
            result = np.expand_dims(result, 1).repeat(number_of_observations, 1)
            p_values += list(result.flatten())

        raw_data["p"] = p_values
        raw_data = raw_data.groupby(["repetition", "client"]).mean().reset_index()
        raw_data.to_csv(cached_path, index=False)
    else:
        print("Using cache...")
        raw_data = pd.read_csv(cached_path)
    print("Extraction completed; plotting data...")
    raw_data.columns = latex_columns
    raw_data = raw_data.round({"dims": 1})
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    sns.lineplot(data=raw_data, x="shift", y=r"$os^*_i$", hue="dims", ax=axes[0],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))
    sns.lineplot(data=raw_data, x="shift", y=r"$p$-val", hue="dims", ax=axes[1],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))

    fig.set_size_inches(figsize)
    move_legend_below_graph(axes, ncol=8)
    axes[-1].set_yscale('log')
    plt.show()


def plot_evaluation_vary_number_pos(filename: str = "synth_po_vary_npo_10reps.csv"):
    # sns.set_palette(sns.color_palette("colorblind", n_colors=4))
    cached_path = os.path.join("results", "cache", filename)
    latex_columns = ["repetition", "client", r"$os^*_i$", r"$os_o$", "shift", "dims", r"\#PO", "is PO", r"$p$-val"]
    if not os.path.exists(cached_path):
        print("Load file...")
        filepath = os.path.join(os.getcwd(), "results", filename)
        raw_data = pd.read_csv(filepath)
        print("File loaded; extracting data...")

        steps = 4
        repetitions_with_same_params = 10
        affected_dims = []
        is_outlier = []
        num_po = []
        for rep in raw_data["repetition"]:
            counter_for_params = int(rep / repetitions_with_same_params)
            affected_dims.append([0.1, 0.3, 0.6, 1.0][int(counter_for_params % steps)])
            num_po.append([0, 1, 3, 6, 10, 20, 25, 30][int(counter_for_params / steps)])

        raw_data["dims"] = affected_dims
        raw_data["#PO"] = num_po
        for i, row in raw_data.iterrows():
            number_of_partition_outliers = num_po[i]
            client = row["client"]
            is_outlier.append(client < number_of_partition_outliers)
        raw_data["is po"] = is_outlier

        raw_data.sort_values(by=["repetition", "client"], inplace=True)
        grouped_by_repetition = raw_data.groupby(by="repetition")
        p_values = []
        for id, group in tqdm(grouped_by_repetition):
            os_federated = []
            number_of_observations = len(group[group["client"] == 0])
            for client, client_data in group.groupby(by="client"):
                os_federated.append(client_data["os_federated"])
                # hacky: we need to adjust labels for the plot! todo: fix it in code
                bool_array = np.logical_and(raw_data["repetition"] == id, raw_data["client"] == client)
                raw_data.loc[bool_array, "labels"] = np.array(group.loc[group["client"] == 0, "labels"])
            result = server_evaluation(os_federated)[1]
            result = np.expand_dims(result, 1).repeat(number_of_observations, 1)
            p_values += list(result.flatten())

        raw_data["p"] = p_values
        raw_data = raw_data.groupby(["repetition", "client"]).mean().reset_index()
        raw_data.to_csv(cached_path, index=False)
    else:
        print("Using cache...")
        raw_data = pd.read_csv(cached_path)
    print("Extraction completed; plotting data...")
    raw_data.columns = latex_columns
    raw_data = raw_data.round({"dims": 1})
    print(raw_data)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    sns.lineplot(data=raw_data, x=r"\#PO", y=r"$os^*_i$", hue="dims", ax=axes[0],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))
    sns.lineplot(data=raw_data, x=r"\#PO", y=r"$p$-val", hue="dims", ax=axes[1],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))

    fig.set_size_inches(figsize)
    move_legend_below_graph(axes, ncol=8)
    axes[-1].set_yscale('log')
    plt.show()


def plot_evaluation_vary_inter_partition_divergence(filename: str = "synth_po_vary_divergence_10reps.csv"):
    cached_path = os.path.join(os.getcwd(), "results", "cache", filename)
    latex_columns = ["repetition", "client", r"$os^*_i$", r"$os_o$", "shift", "dims", "is PO", r"$\Delta$", r"$p$-val"]
    if not os.path.exists(cached_path):
        print("Load file...")
        filepath = os.path.join("results", filename)
        raw_data = pd.read_csv(filepath)
        print("File loaded; extracting data...")

        steps = 10
        affected_dims = []
        repetitions_with_same_params = 10
        variance = []
        for rep in raw_data["repetition"]:
            counter_for_params = int(rep / repetitions_with_same_params)
            affected_dims.append([0.1, 0.3, 0.6, 1.0][int(counter_for_params / steps)])
            variance.append((counter_for_params % steps) / (steps - 1))
        is_outlier = []
        for client in raw_data["client"]:
            is_outlier.append(client == 0)
        raw_data["dims"] = affected_dims
        raw_data["is po"] = is_outlier
        raw_data["variance"] = variance

        raw_data.sort_values(by=["repetition", "client"], inplace=True)
        grouped_by_repetition = raw_data.groupby(by="repetition")
        p_values = []
        for id, group in tqdm(grouped_by_repetition):
            os_federated = []
            number_of_observations = len(group[group["client"] == 0])
            for client, client_data in group.groupby(by="client"):
                os_federated.append(client_data["os_federated"])
                # hacky: we need to adjust labels for the plot! todo: fix it in code
                bool_array = np.logical_and(raw_data["repetition"] == id, raw_data["client"] == client)
                raw_data.loc[bool_array, "labels"] = np.array(group.loc[group["client"] == 0, "labels"])
            result = server_evaluation(os_federated)[1]
            result = np.expand_dims(result, 1).repeat(number_of_observations, 1)
            p_values += list(result.flatten())

        raw_data["p"] = p_values
        raw_data = raw_data.groupby(["repetition", "client"]).mean().reset_index()
        raw_data.to_csv(cached_path, index=False)
    else:
        print("Using cache...")
        raw_data = pd.read_csv(cached_path)
    print("Extraction completed; plotting data...")
    raw_data.columns = latex_columns
    raw_data = raw_data.round({"dims": 1})
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    sns.lineplot(data=raw_data, x=r"$\Delta$", y=r"$os^*_i$", hue="dims", ax=axes[0],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))
    sns.lineplot(data=raw_data, x=r"$\Delta$", y=r"$p$-val", hue="dims", ax=axes[1],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))

    fig.set_size_inches(figsize)
    move_legend_below_graph(axes, ncol=8)
    axes[-1].set_yscale('log')
    plt.show()


def plot_evaluation_vary_num_clients(filename: str = "result.csv"):
    cached_path = os.path.join(os.getcwd(), "results", "cache", filename)
    latex_columns = ["repetition", "client", r"$os^*_i$", r"$os_o$", "shift", r"$\Delta$", "is PO", r"$|N|$",
                     r"$p$-val"]
    if not os.path.exists(cached_path):
        print("Load file...")
        filepath = os.path.join("results", filename)
        raw_data = pd.read_csv(filepath)
        print("File loaded; extracting data...")

        steps = 10
        variance = []
        repetitions_with_same_params = 10
        num_clients = []
        for rep in raw_data["repetition"]:
            counter_for_params = int(rep / repetitions_with_same_params)
            variance.append([0.0, 0.2, 0.4, 0.6][int(counter_for_params % 4)])
            num_clients.append([2, 3, 4, 6, 8, 12, 15, 20, 25, 30][int(counter_for_params / 4)])
        is_outlier = []
        for client in raw_data["client"]:
            is_outlier.append(client == 0)
        raw_data["variance"] = variance
        raw_data["is po"] = is_outlier
        raw_data[r"$|N|$"] = num_clients

        raw_data.sort_values(by=["repetition", "client"], inplace=True)
        grouped_by_repetition = raw_data.groupby(by="repetition")
        p_values = []
        for id, group in tqdm(grouped_by_repetition):
            os_federated = []
            number_of_observations = len(group[group["client"] == 0])
            for client, client_data in group.groupby(by="client"):
                os_federated.append(client_data["os_federated"])
                # hacky: we need to adjust labels for the plot! todo: fix it in code
                bool_array = np.logical_and(raw_data["repetition"] == id, raw_data["client"] == client)
                raw_data.loc[bool_array, "labels"] = np.array(group.loc[group["client"] == 0, "labels"])
            print(result)
            result = server_evaluation(os_federated)[1]
            result = np.expand_dims(result, 1).repeat(number_of_observations, 1)
            p_values += list(result.flatten())

        raw_data["p"] = p_values
        raw_data = raw_data.groupby(["repetition", "client"]).mean().reset_index()
        raw_data.to_csv(cached_path, index=False)
    else:
        print("Using cache...")
        raw_data = pd.read_csv(cached_path)
    print("Extraction completed; plotting data...")
    raw_data.columns = latex_columns
    print(np.unique(raw_data[r"$\Delta$"]))
    raw_data = raw_data.round({"dims": 1})
    raw_data[r"$\Delta$"] = np.round(raw_data[r"$\Delta$"], 1)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    sns.lineplot(data=raw_data, x=r"$|N|$", y=r"$os^*_i$", hue=r"$\Delta$", ax=axes[0],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))
    sns.lineplot(data=raw_data, x=r"$|N|$", y=r"$p$-val", hue=r"$\Delta$", ax=axes[1],
                 style="is PO", palette=sns.cubehelix_palette(n_colors=4))

    fig.set_size_inches(figsize)
    move_legend_below_graph(axes, ncol=8)
    axes[-1].set_yscale('log')
    plt.show()


def plot_roc_curves():
    sns.set_palette(sns.cubehelix_palette(n_colors=3))
    filepath = os.path.join(os.getcwd(), "results", "result.csv")
    raw_data = pd.read_csv(filepath)

    # create labels
    raw_data["shift"] = raw_data["labels"].to_numpy()
    thresh = 96

    def change_label(x):
        client = x["client"]
        shift = x["shift"]
        # Caution: These thresholds come from the way we generate data.
        lower_thresh = 1 / 3
        higher_thresh = 2 / 3
        if shift == -1:
            # inliers
            label = 0
        elif lower_thresh < shift < higher_thresh:
            # global outliers
            label = 2
        else:
            # local outliers
            if 10 <= client < 20:
                label = 0
            else:
                is_local_outlier = (client < 10 and shift >= higher_thresh) or (client >= 20 and shift <= lower_thresh)
                label = 1 if is_local_outlier else 0
        x["labels"] = label
        return x

    # for key, data in raw_data.groupby(by="repetition"):
    #     raw_data[raw_data["repetition"] == key] = data.apply(change_label, axis=1)

    raw_data["global"] = raw_data["labels"] == 2
    raw_data["local"] = raw_data["labels"] == 1
    raw_data["outlier"] = np.logical_or(raw_data["global"], raw_data["local"])

    os_cols = ["os_federated", "os_ondevice", "tandem"]
    titles = ["federated", "local", "tandem"]
    fig, axes = plt.subplots(nrows=1, ncols=len(titles), figsize=figsize)

    def add_precision_recall_plot(os_col: str, ax, title: str = ""):
        result = []
        cache_dir = os.path.join(os.getcwd(), "results", "cache", "df_roc_{}.csv".format(os_col))
        if os.path.exists(cache_dir):
            print("Use cache")
            df = pd.read_csv(cache_dir)
            print(df)
        else:
            for client, data in raw_data.groupby(by="client"):
                if client == 0:
                    continue
                for rep, rep_data in data.groupby(by="repetition"):
                    if os_col == "tandem":
                        # prec_outlier, rec_outlier = tandem_precision_recall_curve(
                        #     rep_data["outlier"],
                        #     probas_pred_local= rep_data["os_ondevice"],
                        #     probas_pred_federated=rep_data["os_federated"],
                        #     thresh_federated=99, pos_labels=[1, 2]
                        # )
                        prec_global, rec_global = tandem_precision_recall_curve(
                            rep_data["global"],
                            probas_pred_local=rep_data["os_ondevice"],
                            probas_pred_federated=rep_data["os_federated"],
                            thresh_federated=thresh, pos_labels=2
                        )
                        prec_local, rec_local = tandem_precision_recall_curve(
                            rep_data["local"],
                            probas_pred_local=rep_data["os_ondevice"],
                            probas_pred_federated=rep_data["os_federated"],
                            thresh_federated=thresh, pos_labels=1
                        )
                        # new_result_outlier = np.vstack([prec_outlier,
                        #                                 rec_outlier,
                        #                                 np.full_like(prec_outlier, fill_value=rep),
                        #                                 np.full(shape=prec_outlier.shape, fill_value="any"),
                        #                                 np.full(shape=prec_outlier.shape, fill_value="tandem"),
                        #                                 np.full(shape=prec_outlier.shape,
                        #                                         fill_value=client)]).T.tolist()
                        new_result_global = np.vstack([prec_global,
                                                       rec_global,
                                                       np.full_like(prec_global, fill_value=rep),
                                                       np.full(shape=prec_global.shape, fill_value="global"),
                                                       np.full(shape=prec_global.shape, fill_value="tandem"),
                                                       np.full(shape=prec_global.shape, fill_value=client)]).T.tolist()

                        new_result_local = np.vstack([prec_local,
                                                      rec_local,
                                                      np.full_like(prec_local, fill_value=rep),
                                                      np.full(shape=prec_local.shape, fill_value="local"),
                                                      np.full(shape=prec_local.shape, fill_value="tandem"),
                                                      np.full(shape=prec_local.shape, fill_value=client)]).T.tolist()
                        result += new_result_global
                        result += new_result_local
                        # result += new_result_outlier
                    else:
                        # prec_outlier, rec_outlier, _ = precision_recall_curve(rep_data["outlier"], probas_pred=rep_data[os_col])
                        prec_local, rec_local, _ = precision_recall_curve(rep_data["local"], probas_pred=rep_data[os_col])
                        prec_global, rec_global, _ = precision_recall_curve(rep_data["global"], probas_pred=rep_data[os_col])
                        new_result_global = np.vstack([prec_global,
                                                       rec_global,
                                                       np.full_like(prec_global, fill_value=rep),
                                                       np.full(shape=prec_global.shape, fill_value="global"),
                                                       np.full(shape=prec_global.shape, fill_value=os_col),
                                                       np.full(shape=prec_global.shape, fill_value=client)]).T.tolist()

                        new_result_local = np.vstack([prec_local,
                                                      rec_local,
                                                      np.full_like(prec_local, fill_value=rep),
                                                      np.full(shape=prec_local.shape, fill_value="local"),
                                                      np.full(shape=prec_local.shape, fill_value=os_col),
                                                      np.full(shape=prec_local.shape, fill_value=client)]).T.tolist()
                        # new_result_outlier = np.vstack([prec_outlier,
                        #                                 rec_outlier,
                        #                                 np.full_like(prec_outlier, fill_value=rep),
                        #                                 np.full(shape=prec_outlier.shape, fill_value="any"),
                        #                                 np.full(shape=prec_outlier.shape, fill_value=os_col),
                        #                                 np.full(shape=prec_outlier.shape, fill_value=client)]).T.tolist()
                        result += new_result_global
                        result += new_result_local
                        # result += new_result_outlie
            result = np.asarray(result)
            df = pd.DataFrame(result, columns=["precision", "recall", "rep", "outlier type", "method", "client"])
            df.to_csv(cache_dir, index=False)
        if title == "tandem":
            print(df[df["outlier type"] == "global"])
        df.dropna(inplace=True, axis=0)
        data = df[df["method"] == os_col]
        sns.lineplot(data=data, x="recall", y="precision", hue="outlier type", ax=ax)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1)

    for ax, os_col, title in zip(axes, os_cols, titles):
        add_precision_recall_plot(os_col, ax, title)

    move_legend_below_graph(axes, 3)

    plt.show()


def move_legend_below_graph(axes, ncol: int):
    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=ncol)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)

    plt.tight_layout()


def create_boxplot():
    sns.set_palette(sns.cubehelix_palette(4))
    df = pd.read_csv(os.path.join("results", "result.csv"))
    df1 = copy.deepcopy(df)
    df2 = copy.deepcopy(df)
    df1["outlier score"] = df["os_federated"]
    df1["detector"] = "federated"
    df2["outlier score"] = df["os_ondevice"]
    df2["detector"] = "local"
    df = pd.concat([df1, df2])
    df["type"] = "inlier"
    df["type"][df["labels"] == 1] = "local outl."
    df["type"][df["labels"] == 2] = "global outl."
    df.sort_values(by="labels", inplace=True)
    ax = sns.boxplot(x="detector", y="outlier score",
                hue="type",
                data=df)
    move_legend_below_graph(axes=np.array([ax]), ncol=4)
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


if __name__ == '__main__':
    plot_evaluation_vary_inter_partition_divergence()
