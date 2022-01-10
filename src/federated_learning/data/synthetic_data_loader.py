import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from torchvision.transforms import transforms


class GaussianMixture():

    def __init__(self, components, dims, seed):
        self.dims = dims
        self.rng = np.random.default_rng(seed)
        self.centers = self.rng.uniform(low=0.2, high=0.8, size=(components, dims))
        self.pattern_std = 0.1
        self.own_patterns = np.random.choice(np.arange(components), int(components / 2.0), replace=False)
        self.other_patterns = np.array([i for i in np.arange(components) if i not in self.own_patterns])

    def sample_from_own(self, n):
        d = []
        for i in range(n):
            pattern = np.random.choice(self.own_patterns)
            mean = self.centers[pattern]
            d.append(centered_truncnorm(2, mean, self.pattern_std, size=self.dims))
        return np.array(d)

    def sample_from_other(self, n):
        d = []
        for i in range(n):
            pattern = np.random.choice(self.other_patterns)
            mean = self.centers[pattern]
            d.append(centered_truncnorm(2, mean, self.pattern_std, size=self.dims))
        return np.array(d)

    def sample_random(self, n):
        return np.random.uniform(size=(n, self.dims))


def centered_truncnorm(width_std, mean, std, size):
    my_mean = mean
    my_std = std
    myclip_a = my_mean - width_std * my_std
    myclip_b = my_mean + width_std * my_std
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def make_partition_outlier(data, destination, shift_amount, fraction: float = 1.0) -> pd.DataFrame:
    """
    shift from a to b by shift amount
    :param data: the data to shift
    :param shift_amount: the amount to shift between a and b.
    :return: the data
    """
    start = data.mean(axis=0)
    vector = destination - start
    dims = data.shape[1]
    num_shifted_dims = int(fraction*dims)
    random_dims = np.random.choice(np.arange(dims), replace=False, size=num_shifted_dims)
    data[:, random_dims] = data[:, random_dims] + vector[random_dims] * shift_amount
    return data


def get_data_for_index(index: int, nobs: int, nout: int = 0, seed: int = 0):
    dims = 20
    rng = np.random.default_rng(seed)
    means_a = rng.uniform(-70, -30, size=dims) / 100.0
    means_b = rng.uniform(30, 70, size=dims) / 100.0
    means_a = means_a + centered_truncnorm(2, 0.0, 0.01, dims)
    means_b = means_b + centered_truncnorm(2, 0.0, 0.01, dims)  # add noise
    num_steps_outliers = 20
    scale = 0.02

    def create_outlier_line(nout: int, start_loc: np.ndarray, end_loc: np.ndarray):
        if nout == 0:
            return np.empty(shape=(nout, dims), dtype=float)
        start = start_loc
        end = end_loc
        line = np.array([
            start + (end-start) * float(i) / nout
            for i in range(nout)
        ])
        noise = centered_truncnorm(2, 0, scale, line.shape)  # add some small noise to line
        return line + noise

    def sample_dist_a(size, scale=0.15):
        return centered_truncnorm(2, means_a, scale, size)

    def sample_dist_b(size, scale=0.15):
        return centered_truncnorm(2, means_b, scale, size)

    if index < 10:
        data_1 = sample_dist_a(size=(2*nobs, dims), scale=scale)
        outliers = create_outlier_line(nout, means_a, means_b)
        data = np.vstack([data_1, outliers])
        labels = np.concatenate([-np.ones(data_1.shape[0]),
                                 np.array([i/num_steps_outliers for i in range(num_steps_outliers+1)])])
    elif 10 <= index < 20:
        data_1 = sample_dist_a(size=(nobs, dims), scale=scale)
        data_2 = sample_dist_b(size=(nobs, dims), scale=scale)
        outliers = create_outlier_line(nout, means_a, means_b)
        data = np.vstack([data_1, data_2, outliers])
        labels = np.concatenate([-np.ones(data_1.shape[0]),
                                 -np.ones(data_2.shape[0]),
                                 np.array([i / num_steps_outliers for i in range(num_steps_outliers + 1)])])
    else:
        data_2 = sample_dist_b(size=(nobs, dims), scale=scale)
        outliers = create_outlier_line(nout, means_a, means_b)
        data = np.vstack([data_2, outliers])
        labels = np.concatenate([-np.ones(data_2.shape[0]),
                                 np.array([i / num_steps_outliers for i in range(num_steps_outliers + 1)])])
    return (data + 1.0) / 2.0, labels.astype(str)


def get_data_for_index_2(index: int, nobs: int, rep: int, nout: int = 0, seed: int = 0,
                         return_gmm: bool = False, is_po: bool = False, deviation: float = 0.0):
    dims = 10
    share_global = (rep % 6) / 5.0
    nout_total = 2 * nout
    nout_local = int((1 - share_global) * nout_total)
    nout_global = int(share_global * nout_total)
    print(rep, nout_local, nout_global)
    gmm = GaussianMixture(components=10, dims=dims, seed=seed)
    inliers = gmm.sample_from_own(nobs)
    local_out = gmm.sample_from_other(nout_local)
    global_out = gmm.sample_random(nout_global)

    if is_po:
        inliers = add_po(inliers, deviation=deviation, bias=True)

    if nout_local > 0 and nout_global > 0:
        data = np.vstack([inliers, local_out, global_out])
        labels = np.concatenate([np.zeros(shape=len(inliers)),
                                 np.ones(shape=len(local_out)),
                                 np.ones(shape=len(global_out)) * 2])
    elif nout_local > 0 and nout_global == 0:
        data = np.vstack([inliers, local_out])
        labels = np.concatenate([np.zeros(shape=len(inliers)),
                                 np.ones(shape=len(local_out))])
    elif nout_local == 0 and nout_global > 0:
        data = np.vstack([inliers, global_out])
        labels = np.concatenate([np.zeros(shape=len(inliers)),
                                 np.ones(shape=len(global_out)) + 1])
    else:
        data = inliers
        labels = np.zeros(shape=len(data))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    if return_gmm:
        return data[indices], labels[indices].astype(str), gmm
    else:
        return data[indices], labels[indices].astype(str)


def add_po(data: np.ndarray, deviation: float, bias: bool = True):
    if bias:
        bias_vector = np.random.uniform(0, deviation, size=data.shape)
        is_negative = np.sign(data) == -1
        bias_vector[is_negative] = bias_vector[is_negative] * -1
        return data + bias_vector


class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform, labels):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.from_numpy(sample)
        sample = torch.unsqueeze(sample, 0)
        return sample, label


def __synthetic_data_loader__(index, nobs: int, nout: int, rep: int, batch_size: int = 32, seed: int = 0,
                              deviation: float = 0.0, is_po: bool = False):
    data, labels = get_data_for_index_2(index, nobs=nobs, nout=nout, rep=rep, seed=seed,
                                        is_po=is_po, deviation=deviation)
    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    print(data)
    dataset = SyntheticDataset(data=data, transform=base_transform, labels=labels)
    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False))


def __synthetic_data_loader_with_partition_outlier__(index, random_seed: int, partition_outlier: bool, nobs: int,
                                                     shift_amount: float, fraction: float, batch_size: int = 64):
    data, labels = get_data_for_index(index, random_seed=random_seed, nout=0, nobs=nobs)
    mean_b = 0.4 * np.ones(data.shape[-1])
    if partition_outlier:
        data = make_partition_outlier(data, destination=mean_b, shift_amount=shift_amount, fraction=fraction)
    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    dataset = SyntheticDataset(data=data, transform=base_transform, labels=labels)
    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False))


def normal_data_loader_with_partition_outlier(nobs: int, dims: int, fraction_dims: float,
                                              shift_amount: float, batch_size: int = 32, std_divergence: float = 0.0):
    data = np.random.normal(size=(nobs, dims))
    noise = np.random.uniform(low=-std_divergence, high=std_divergence, size=dims)
    data = data + noise
    affected_dims = np.random.choice(np.arange(dims), int(fraction_dims*dims))
    shift_direction_signs = np.random.choice([-1, 1], size=len(affected_dims))
    shift = shift_amount * shift_direction_signs
    data[:, affected_dims] = data[:, affected_dims] + shift
    labels = ["{}".format(shift_amount) for _ in range(nobs)]

    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    dataset = SyntheticDataset(data=data, transform=base_transform, labels=labels)
    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False))


def load_synthetic_partition(index: int, *args, **kwargs):
    if "experiment" not in kwargs:
        raise KeyError("Key `experiment` should be in kwargs.")
    experiment = kwargs["experiment"]
    rep = int(kwargs["exp_index"])
    nobs = int(kwargs["nobs"])
    if experiment == "local/global":
        return __synthetic_data_loader__(index, rep=rep, nout=21, nobs=nobs, seed=rep, is_po=False)
    if experiment == "partition_outlier":
        print("Experiment partition_outlier")
        is_po = index < 1
        deviation = 3 * rep / 9.0
        return __synthetic_data_loader__(index, nobs=nobs, nout=0, rep=rep, seed=rep, is_po=is_po, deviation=deviation)

