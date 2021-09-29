import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def get_data_for_index(index: int, random_seed: int, nout: int = 0):
    nobs = 2000
    dims = 20
    means_a = -0.4 * np.ones(dims)
    means_b = 0.4 * np.ones(dims)
    if index != 0:
        means_a = means_a + stats.truncnorm.rvs(-0.1, 0.1, loc=0.0, scale=0.03, size=dims)  # add noise
        means_b = means_b + stats.truncnorm.rvs(-0.1, 0.1, loc=0.0, scale=0.03, size=dims)  # add noise
    num_steps_outliers = 20
    scale = 0.5 / 2

    def create_outlier_line(nout: int, start_loc: np.ndarray, end_loc: np.ndarray):
        if nout == 0:
            return np.empty(shape=(nout, dims), dtype=float)
        start = start_loc
        end = end_loc
        return np.array([
            start + (end-start) * float(i) / nout
            for i in range(nout)
        ])

    def sample_dist_a(size, scale=0.05):
        return stats.truncnorm.rvs(-3*scale, 3*scale, loc=means_a, scale=scale, size=size)

    def sample_dist_b(size, scale=0.05):
        return stats.truncnorm.rvs(-2*scale, 2*scale, loc=means_b, scale=scale, size=size)

    if index < 10:
        data_1 = sample_dist_a(size=(2*nobs, dims), scale=scale)
        outliers = create_outlier_line(nout, means_a, means_b)
        data = np.vstack([data_1, outliers])
        labels = np.concatenate([-np.ones(data_1.shape[0]),
                                 np.array([i/num_steps_outliers for i in range(num_steps_outliers+1)])])
        return data, labels.astype(str)
    elif 10 <= index < 20:
        data_1 = sample_dist_a(size=(nobs, dims), scale=scale)
        data_2 = sample_dist_b(size=(nobs, dims), scale=scale)
        outliers = create_outlier_line(nout, means_a, means_b)
        data = np.vstack([data_1, data_2, outliers])
        labels = np.concatenate([-np.ones(data_1.shape[0]),
                                 -np.ones(data_2.shape[0]),
                                 np.array([i / num_steps_outliers for i in range(num_steps_outliers + 1)])])
        return data, labels.astype(str)
    else:
        data_2 = sample_dist_b(size=(nobs, dims), scale=scale)
        outliers = create_outlier_line(nout, means_a, means_b)
        data = np.vstack([data_2, outliers])
        labels = np.concatenate([-np.ones(data_2.shape[0]),
                                 np.array([i / num_steps_outliers for i in range(num_steps_outliers + 1)])])
        return data, labels.astype(str)


def make_partition_outlier(data, destination, shift_amount) -> pd.DataFrame:
    """
    shift from a to b by shift amount
    :param data: the data to shift
    :param shift_amount: the amount to shift between a and b.
    :return: the data
    """
    start = data.mean(axis=0)
    vector = destination - start
    print(vector)
    return data + vector*shift_amount


def __synthetic_data_loader_with_partition_outlier__(index, random_seed: int, partition_outlier: bool,
                                                     shift_amount: float, batch_size: int = 64):
    data, labels = get_data_for_index(index, random_seed=random_seed, nout=0)
    mean_b = 0.4 * np.ones(data.shape[-1])
    if partition_outlier:
        data = make_partition_outlier(data, destination=mean_b, shift_amount=shift_amount)
    return data


if __name__ == '__main__':
    data_0 = __synthetic_data_loader_with_partition_outlier__(0, random_seed=0, partition_outlier=True, shift_amount=0.0)
    data_05 = __synthetic_data_loader_with_partition_outlier__(0, random_seed=0, partition_outlier=True,
                                                               shift_amount=0.5)
    data_1 = __synthetic_data_loader_with_partition_outlier__(0, random_seed=0, partition_outlier=True,
                                                              shift_amount=1.0)
    data_a = __synthetic_data_loader_with_partition_outlier__(1, random_seed=0, partition_outlier=False, shift_amount=0.0)
    data_ab = __synthetic_data_loader_with_partition_outlier__(10, random_seed=0, partition_outlier=False, shift_amount=0.0)
    data_b = __synthetic_data_loader_with_partition_outlier__(20, random_seed=0, partition_outlier=False, shift_amount=0.0)

    data = [data_a, data_ab, data_b, data_0, data_05, data_1]
    for i, d in enumerate(data):
        indices = np.arange(len(d))
        choice = np.random.choice(indices, 100, replace=False)
        choice = d[choice]
        plt.scatter(choice[:, 0], choice[:, 1], label="{}".format(i))
    plt.legend()
    plt.show()
