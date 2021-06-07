import numpy as np
import torch
from sklearn import preprocessing
import scipy.stats as stats
from torchvision.transforms import transforms


def get_data_for_index(index: int, random_seed: int):
    nobs = 2000
    dims = 20
    means_a = -0.5*np.ones(dims)
    means_b = 0.5*np.ones(dims)
    num_steps_outliers = 20
    scale = 0.5

    def create_outlier_line(nout: int, start_loc: np.ndarray, end_loc: np.ndarray):
        scale = 0.05  # add some random noise to outliers
        start = stats.truncnorm.rvs(start_loc - 3*scale, start_loc + 3*scale, loc=start_loc, scale=scale)
        end = stats.truncnorm.rvs(end_loc - 3*scale, end_loc + 3*scale, loc=end_loc, scale=scale)
        return np.array([
            start + (end-start) * float(i) / nout + np.random.normal(scale=1/3 * 1/nout)
            for i in range(nout)
        ])

    def sample_dist_a(size, scale=0.05):
        return stats.truncnorm.rvs(means_a - scale, means_a + scale, loc=means_a, scale=scale, size=size)

    def sample_dist_b(size, scale=0.05):
        return stats.truncnorm.rvs(means_b - scale, means_b + scale, loc=means_b, scale=scale, size=size)

    if index < 10:
        data_1 = sample_dist_a(size=(2*nobs, dims), scale=scale)
        outliers = create_outlier_line(21, means_a, means_b)
        data = np.vstack([data_1, outliers])
        labels = np.concatenate([np.zeros(data_1.shape[0]),
                                 np.array([i/num_steps_outliers for i in range(num_steps_outliers+1)])])
        return data, labels.astype(str)
    elif 10 <= index < 20:
        data_1 = sample_dist_a(size=(nobs, dims), scale=scale)
        data_2 = sample_dist_b(size=(nobs, dims), scale=scale)
        outliers = create_outlier_line(21, means_a, means_b)
        data = np.vstack([data_1, data_2, outliers])
        labels = np.concatenate([np.zeros(data_1.shape[0]),
                                 np.ones(data_2.shape[0]),
                                 np.array([i / num_steps_outliers for i in range(num_steps_outliers + 1)])])
        return data, labels.astype(str)
    else:
        data_2 = sample_dist_b(size=(nobs, dims), scale=scale)
        outliers = create_outlier_line(21, means_a, means_b)
        data = np.vstack([data_2, outliers])
        labels = np.concatenate([np.ones(data_2.shape[0]),
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
    data, labels = get_data_for_index(index, random_seed=random_seed)
    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    dataset = SyntheticDataset(data=data, transform=base_transform, labels=labels)
    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False))


def load_synthetic_partition(index: int):
    return __synthetic_data_loader__(index, random_seed=0)
