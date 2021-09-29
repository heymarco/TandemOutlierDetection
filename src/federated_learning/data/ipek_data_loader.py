import logging
import os
import sys
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from scipy.stats import zscore
from sklearn import preprocessing
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, ".")
from src.helper import extract_features_in_sliding_window, ipek_split_ratios


def extract_features(window_size: int = 20, stride: int = 20):
    data_dir = os.path.join("data", "ipek")
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, sep=";")
        file.drop(["time"], axis=1, inplace=True)
        file = file[file["batC"] > 0.6]
        file = file.diff().dropna()
        file.reset_index(drop=True, inplace=True)
        features = extract_features_in_sliding_window(file, window_size=window_size, stride=stride)
        feature_file_path = os.path.join(data_dir, "features", filename)
        features.to_csv(feature_file_path, sep=";")


def plot_ipek():
    data_dir = os.path.join("data", "ipek")
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    filenames = os.listdir(data_dir)
    filenames = sorted(filenames)
    fig, axes = plt.subplots(9, 1, sharex="all")
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, sep=";")
        file.drop(["time"], axis=1, inplace=True)
        file = file[file["batC"] > 0.6]
        file.reset_index(drop=True, inplace=True)
        for index, col in enumerate(file.columns):
            axes.flatten()[index].plot(file[col])
            axes.flatten()[index].set_ylabel(col)
        break
    plt.tight_layout()
    plt.show()


class IpekRawDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform, labels=False, seq_len=20, jump=10):
        self.data = data
        self.transform = transform
        self.labels = labels
        self.seq_len = seq_len
        self.jump = jump

    def __len__(self):
        return int((len(self.data) - self.seq_len) / self.jump)

    def __getitem__(self, idx):
        sample = self.data[idx*self.jump:(idx*self.jump)+self.seq_len]
        sample = torch.from_numpy(sample)
        return sample, "unlabeled"


class IpekDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform, labels=False):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.from_numpy(sample)
        sample = torch.unsqueeze(sample, 0)
        return sample, "unlabeled"


def __load_raw_data(index: int, seq_len=100):
    log(logging.INFO, "Loading data for device {} ...".format(index))
    data_dir = os.path.join("data", "ipek")
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    filenames = os.listdir(data_dir)
    filenames = sorted(filenames)
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        right_file = filename.endswith("P{}.csv".format(index + 1))
        if not right_file:
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, sep=";", index_col=0)
        file.drop(["time"], axis=1, inplace=True)
        file = file[file["batC"] > 0.6]
        file = file.diff().dropna()
        data_columns = file.columns

        # if index < 15:
        #     file = pd.concat([
        #         file.iloc[int(len(file) * ipek_split_ratios[index]):],  # Schrauben
        #         # file.iloc[:int(len(file) * ipek_split_ratios[index])].sample(2),
        #     ])
        # else:
        #     file = pd.concat([
        #         file.iloc[:int(len(file) * ipek_split_ratios[index])],  # Bohren
        #         # file.iloc[int(len(file) * ipek_split_ratios[index]):].sample(2),
        #     ])
        x = file[data_columns].values  # returns a numpy array
        x_scaled = StandardScaler().fit_transform(x)
        x_scaled[np.isnan(x_scaled)] = 0.0
        x_scaled = x_scaled[:len(x_scaled)-(len(x_scaled) % seq_len)]
        data_normed = pd.DataFrame(x_scaled, columns=data_columns)
        data_normed.reset_index(drop=True, inplace=True)
        return data_normed


def __load_data__(index: int):
    log(logging.INFO, "Loading data for device {} ...".format(index))
    data_dir = os.path.join("data", "ipek", "features")
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    filenames = os.listdir(data_dir)
    filenames = sorted(filenames)
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        right_file = filename.endswith("P{}.csv".format(index+1))
        if not right_file:
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, sep=";", index_col=0)
        data_columns = file.columns

        # if index < 15:
        #     file = pd.concat([
        #         file.iloc[int(len(file) * ipek_split_ratios[index]):],  # Schrauben
        #         # file.iloc[:int(len(file) * ipek_split_ratios[index])].sample(2),
        #     ])
        # else:
        #     file = pd.concat([
        #         file.iloc[:int(len(file) * ipek_split_ratios[index])],  # Bohren
        #         # file.iloc[int(len(file) * ipek_split_ratios[index]):].sample(2),
        #     ])
        x = file[data_columns].values  # returns a numpy array
        x_scaled = StandardScaler().fit_transform(x)
        x_scaled[np.isnan(x_scaled)] = 0.0
        data_normed = pd.DataFrame(x_scaled, columns=data_columns)
        data_normed.reset_index(drop=True, inplace=True)
        return data_normed


def __ipek_data_loader__(index, batch_size: int = 32):
    data = __load_data__(index)
    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    dataset = IpekDataset(data=data.to_numpy(), transform=base_transform)
    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True))


def __ipek_raw_loader__(index, batch_size: int = 32):
    data = __load_raw_data(index)
    base_transforms_list = [transforms.ToTensor()]
    base_transform = transforms.Compose(base_transforms_list)
    dataset = IpekRawDataset(data=data.to_numpy(), transform=base_transform)
    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True))


def load_ipek_partition(index: int, *args, **kwargs):
    return __ipek_data_loader__(index)


if __name__ == '__main__':
    extract_features(window_size=100, stride=25)
