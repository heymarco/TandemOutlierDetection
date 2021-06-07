import os
import sys

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, ".")
from src.helper import server_evaluation


if __name__ == '__main__':
    filepath = os.path.join("results", "results.csv")
    raw_data = pd.read_csv(filepath)

    grouped_by_exp_index = raw_data.groupby(by="repetition")
    for tpl in grouped_by_exp_index:
        df = tpl[1]
        exp_results = []
        grouped_by_client = df.groupby(by="client")
        os_federated = [
            data[1]["os_federated"].to_numpy() for data in grouped_by_client
        ]
        os_star, probabilities = server_evaluation(os_federated)

        print(os_star)
        print(probabilities)

        for i, outlier_scores in enumerate(os_federated):
            sns.kdeplot(outlier_scores, label="Device {}".format(i+1), alpha=0.3)
        plt.legend()
        plt.show()
        plt.clf()

        alpha = 0.05
        indices = np.arange((len(probabilities)))
        x_inlier = indices[probabilities > alpha]
        x_outlier = indices[probabilities <= alpha]
        y_inlier = probabilities[probabilities > alpha]
        y_outlier = probabilities[probabilities <= alpha]

        plt.scatter(x_inlier, y_inlier, marker="s", color="black")
        plt.scatter(x_outlier, y_outlier, marker="s", color="red")
        plt.axhline(0.05, color="black", label="alpha=0.05", lw=0.5)
        plt.ylabel("p-value")
        plt.xlabel("device index")
        plt.tight_layout()
        plt.legend()
        plt.show()
