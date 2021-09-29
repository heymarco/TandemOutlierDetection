import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import keras
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def get_data_for_index(x: np.ndarray, y: np.ndarray, i: int, contamination: float = 0.01):
    y_normal_indices = np.array([index for index, label in enumerate(y) if label < 9 and label != i], dtype=int)
    y_normal = np.zeros(len(y_normal_indices), dtype=int)
    x_normal = x[y_normal_indices]

    local_outlier_indices = [index for index, label in enumerate(y) if label == i]
    global_outlier_indices = [index for index, label in enumerate(y) if label == 9]

    local_outlier_indices = np.random.choice(local_outlier_indices,
                                             int(len(x_normal) * contamination),
                                             replace=False)
    global_outlier_indices = np.random.choice(global_outlier_indices,
                                              int(len(x_normal) * contamination),
                                              replace=False)

    x_local = x[local_outlier_indices]
    y_local = np.ones(len(local_outlier_indices), dtype=int)
    x_global = x[global_outlier_indices]
    y_global = np.ones(len(global_outlier_indices), dtype=int) * 2

    data = np.concatenate([x_normal, x_local, x_global])
    labels = np.concatenate([y_normal, y_local, y_global])

    return data, labels.astype(str)


class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform, labels):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].flatten()
        label = self.labels[idx]
        sample = sample / np.max(sample)
        sample = torch.from_numpy(sample)
        sample = torch.unsqueeze(sample, 0)
        return sample, label


def __mnist_data_loader__(batch_size: int = 64, random_seed: int = 0, num_clients: int = 8):
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_full = x_train
    y_full = y_train

    shuffled_indices = np.array([i for i in range(len(y_full)) if y_full[i] != 8])
    np.random.RandomState(random_seed).shuffle(shuffled_indices)

    x = x_full[shuffled_indices]
    y = y_full[shuffled_indices]

    num_data_per_client = int(len(x) / num_clients)
    x_split = [x[i * num_data_per_client: (i + 1) * num_data_per_client] for i in range(num_clients)]
    y_split = [y[i * num_data_per_client: (i + 1) * num_data_per_client] for i in range(num_clients)]
    federated_data = [get_data_for_index(x_split[i], y_split[i], i) for i in range(num_clients)]

    def make_partition_outlier(x, y, x_full, y_full, label, percentage):
        po_indices = [i for i in range(len(y_full)) if y_full[i] == label]
        num_changes = int(len(y)*percentage)
        po_indices = np.random.RandomState(random_seed).choice(po_indices, num_changes, replace=False)
        x[:num_changes] = x_full[po_indices]
        y[:num_changes] = y_full[po_indices]
        return x, y

    federated_data[-1] = make_partition_outlier(federated_data[-1][0], federated_data[-1][1], x_full, y_full,
                                                num_clients-1, 0.3)

    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    datasets = [MnistDataset(data=tpl[0], transform=base_transform, labels=tpl[1])
                for tpl in federated_data]
    return [(torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True),
             torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)) for ds in datasets]


def load_mnist_partition(index: int, *args, **kwargs):
    return __mnist_data_loader__()[index]
