import numpy as np
import torch
from torchvision.transforms import transforms

from src.helper import centered_truncnorm


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


def get_data_for_index_2(nobs: int, rep: int, nout: int = 0, seed: int = 0,
                         return_gmm: bool = False, is_po: bool = False, deviation: float = 0.0):
    dims = 10
    share_global = (rep % 6) / 5.0
    nout_total = 2 * nout
    nout_local = int((1 - share_global) * nout_total)
    nout_global = int(share_global * nout_total)
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
        bias_vector = np.ones_like(data) * deviation
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
    data, labels = get_data_for_index_2(nobs=nobs, nout=nout, rep=rep, seed=seed,
                                        is_po=is_po, deviation=deviation)
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
        is_po = index < 1
        tested_nobs = [10000, 1000, 100]
        nobs = tested_nobs[int(rep / 50)]
        deviation = (rep % 5) / 4.0
        pattern_std = 0.1
        deviation_std = deviation * pattern_std
        return __synthetic_data_loader__(index, nobs=nobs, nout=0, rep=rep, seed=rep,
                                         is_po=is_po, deviation=deviation_std)

