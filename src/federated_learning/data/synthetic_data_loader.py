import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
import scipy.stats as stats
from torchvision.transforms import transforms


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


def get_data_for_index(index: int, random_seed: int, nobs: int, nout: int = 0):
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
        return stats.truncnorm.rvs(-2*scale, 2*scale, loc=means_a, scale=scale, size=size)

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


def __synthetic_data_loader__(index, random_seed: int, batch_size: int = 64):
    data, labels = get_data_for_index(index, random_seed=random_seed, nout=21)
    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
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
    if experiment == "local/global":
        return __synthetic_data_loader__(index, random_seed=0)
    if experiment == "partition outlier":
        rep = float(kwargs["rep"])
        fraction = float(kwargs["fraction"])
        shift = float(kwargs["shift"])
        dims = int(kwargs["dims"])
        nobs = int(kwargs["nobs"])
        is_po = bool(kwargs["is_po"])
        inter_client_variance = float(kwargs["inter_client_variance"])
        if is_po:
            print("Rep {}: {} shift".format(rep, shift))
        return normal_data_loader_with_partition_outlier(nobs=nobs, dims=dims,
                                                         fraction_dims=fraction, shift_amount=shift,
                                                         std_divergence=inter_client_variance)
        # return __synthetic_data_loader_with_partition_outlier__(index=index, random_seed=0,
        #                                                         nobs=nobs, fraction=fraction,
        #                                                         partition_outlier=is_po,
        #                                                         shift_amount=shift_amount)

