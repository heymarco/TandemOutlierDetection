import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, ".")
from src.helper import get_point_outliers

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')
sns.set_palette(sns.color_palette("rocket"))


def plot_outlier_scores(thresh: float = 99.0):
    nrows = 5
    ncols = 6
    fig_o, axes_o = plt.subplots(nrows, ncols, sharey="all")
    fig_f, axes_f = plt.subplots(nrows, ncols, sharey="all")
    filepath = os.path.join("results", "results.csv")
    raw_data = pd.read_csv(filepath)
    grouped_data = raw_data.groupby(by="client")
    fig_o.suptitle("Ondevice")
    fig_f.suptitle("Federated")
    percentile = thresh
    for i, (key, client_data) in enumerate(grouped_data):
        lo, go, id = get_point_outliers(os_ondevice=client_data["os_ondevice"],
                                        os_federated=client_data["os_federated"],
                                        percentile=percentile)
        oso = client_data["os_ondevice"]
        x_lo = [i for i in range(len(lo)) if lo[i]]
        x_go = [i for i in range(len(lo)) if go[i]]
        x_id = [i for i in range(len(lo)) if id[i]]
        axes_o.flatten()[i].scatter(range(len(oso)), oso, marker=".", alpha=0.2)
        axes_o.flatten()[i].scatter(x_lo, oso[lo], marker=".", alpha=0.5)
        axes_o.flatten()[i].scatter(x_go, oso[go], marker=".", alpha=0.5)
        axes_o.flatten()[i].scatter(x_id, oso[id], marker=".", alpha=0.5)
        axes_o.flatten()[i].axhline(np.percentile(oso, percentile), color="black", ls="dotted")
        osf = client_data["os_federated"]
        axes_f.flatten()[i].scatter(range(len(oso)), osf, marker=".", alpha=0.2)
        axes_f.flatten()[i].scatter(x_lo, osf[lo], marker=".", alpha=0.5)
        axes_f.flatten()[i].scatter(x_go, osf[go], marker=".", alpha=0.5)
        axes_f.flatten()[i].scatter(x_id, osf[id], marker=".", alpha=0.5)
        print(np.percentile(oso, percentile))
        print(np.percentile(osf, percentile))
        axes_f.flatten()[i].axhline(np.percentile(osf, percentile), color="black", ls="dotted")
    plt.show()


def plot_outlier_scores_over_distance(thresh: float = 99.0):
    sns.set_palette(sns.color_palette("colorblind", n_colors=2))
    nrows = 1
    ncols = 3
    fig_o, axes_o = plt.subplots(nrows, ncols, sharey="all", sharex="all")
    filepath = os.path.join("results", "results.csv")
    raw_data = pd.read_csv(filepath)

    def plot_data(ax, data, title):
        percentile = thresh
        data = data[data["repetition"] == 0]
        sns.lineplot(data=data, x="labels", y="os_ondevice", ax=ax, ci="sd", label=r"$os_o$")
        sns.lineplot(data=data, x="labels", y="os_federated", ax=ax, ci="sd", label=r"$os_f$")
        percentile_o = np.percentile(data["os_ondevice"], percentile)
        percentile_f = np.percentile(data["os_federated"], percentile)
        print(percentile_o)
        print(percentile_f)
        ax.axhline(percentile_o, color=ax.get_lines()[0].get_color(), ls="dotted", label=r"${}\%\ os_o$".format(percentile))
        ax.axhline(percentile_f, color=ax.get_lines()[1].get_color(), ls="dotted", label=r"${}\%\ os_f$".format(percentile))
        ax.set_title(title)

    # first set of devices
    relevant_data = raw_data[raw_data["client"] < 10]
    plot_data(axes_o.flatten()[0], relevant_data, "Task A")
    # second set of devices
    relevant_data = raw_data[np.logical_and(raw_data["client"] >= 10, raw_data["client"] < 20)]
    plot_data(axes_o.flatten()[1], relevant_data, "Task A and B")
    # third set of devices
    relevant_data = raw_data[raw_data["client"] >= 20]
    plot_data(axes_o.flatten()[2], relevant_data, "Task B")
    handles, labels = axes_o.flatten()[-1].get_legend_handles_labels()
    for ax in axes_o.flatten():
        ax.get_legend().remove()
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["A", "inbetween", "B"])
        ax.set_xlabel("")
        ax.set_ylabel(r"$os_{o}, os_{f}$")
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=4)
    plt.show()


if __name__ == '__main__':
    plot_outlier_scores_over_distance()
