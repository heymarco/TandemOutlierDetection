import os
import sys

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, ".")
from src.helper import server_evaluation

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')
# sns.set_palette(sns.color_palette("rocket"))


def plot_partition_outlier():
    filepath = os.path.join("results", "ipek_results.csv")
    raw_data = pd.read_csv(filepath)
    colors = sns.color_palette("vlag_r", as_cmap=False, n_colors=1200)
    colors = colors[:500] + colors[700:]
    # colors = [colors[-i] for i in range(len(colors))]

    grouped_by_exp_index = raw_data.groupby(by="repetition")
    for tpl in grouped_by_exp_index:
        df = tpl[1]
        exp_results = []
        grouped_by_client = df.groupby(by="client")
        os_federated = [
            data[1]["os_federated"].to_numpy() for data in grouped_by_client
        ]
        os_star, probabilities = server_evaluation(os_federated)

        alpha = 0.05
        indices = np.arange((len(probabilities)))
        x_inlier = indices[probabilities > alpha]
        x_outlier = indices[probabilities <= alpha]
        y_inlier = probabilities[probabilities > alpha]
        y_outlier = probabilities[probabilities <= alpha]

        fig, axes = plt.subplots(2, 1, figsize=(4, 4))

        for i, outlier_scores in enumerate(os_federated):
            color = colors[int(np.floor(probabilities[i]*1000))]
            _alpha = 0.7
            sns.kdeplot(outlier_scores, label="Device {}".format(i+1), alpha=_alpha, color=color,
                        ax=axes[0], zorder=int(100*(1-probabilities[i])))
        axes[0].set_ylabel("kde")
        axes[0].set_xlabel("outlier score")

        for i, prob in enumerate(probabilities):
            color = colors[int(np.floor(probabilities[i] * 1000))]
            axes[1].scatter([i], [prob], color=color)
        axes[1].axhline(0.05, color="black", label=r"$\alpha=0.05$", lw=0.5)
        axes[1].set_ylabel("p-value")
        axes[1].set_xlabel("device index")
        ticks = [index for index in range(len(probabilities)) if index % 2]
        labels = ["{}".format(int(index+1)) for index in ticks]
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels(labels)
        axes[1].set_yticks([0.0, 0.5, 1.0])
        axes[1].set_yticklabels(["0.0", "0.5", "1.0"])
        plt.tight_layout()
        plt.legend()

        handles, labels = axes[0].get_legend_handles_labels()
        alpha_handles, alpha_labels = axes[1].get_legend_handles_labels()

        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()
        handles = [
            handles[np.argmax(probabilities)],
            handles[np.argmin(probabilities)],
            alpha_handles[0]
        ]
        labels = [
            "Normal",
            "Outliers (devices {})".format([i + 1 for i in range(len(probabilities))
                                                      if probabilities[i] <= alpha]),
            alpha_labels[0]
        ]
        plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=3)

        plt.show()



if __name__ == '__main__':
    plot_partition_outlier()
