import os
import logging
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler


def load_data_create_partitions(index: int) -> pd.DataFrame:
    log(logging.INFO, "Loading data for device {} ...".format(index))
    data_dir = os.path.join("data", "arculus")
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        if not filename.endswith("100_flattened.csv"):
            continue
        right_file = filename.startswith("arculee5-30") if index == 19 else filename.startswith("arculee5-29")
        if not right_file:
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, index_col=0, sep=",")
        device_index_col = [index for _ in range(len(file))]
        file["device_index"] = device_index_col
        file.reset_index(drop=True, inplace=True)
        file_length = len(file)
        clean_data_max_index = int(file_length / 2)
        clean_data_max_index = clean_data_max_index - clean_data_max_index % 19
        num_data_per_client = int(clean_data_max_index / 19)
        log(logging.INFO, "Done loading data for client {}.".format(index))
        if index == 19:
            file = file.iloc[-num_data_per_client:]
        else:
            file = file.iloc[index*num_data_per_client:(index+1)*num_data_per_client]
        file.reset_index(drop=True, inplace=True)
        print(file.index)
        print(file.columns)
        file.drop(["timestamp"], axis=1, inplace=True)
        return file


def load_data(index: int) -> pd.DataFrame:
    log(logging.INFO, "Loading data for device {} ...".format(index))
    data_dir = os.path.join("data", "arculus")
    file_indices = {
        1: "arculee5-19",
        2: "arculee5-29",
        3: "arculee5-30"
    }
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        if not filename.endswith("100_flattened.csv"):
            continue
        if not filename.startswith(file_indices[index]):
            continue
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file, index_col=0, sep=",")
        device_index_col = [index for _ in range(len(file))]
        file["device_index"] = device_index_col
        log(logging.INFO, "Done loading data.")
        file.reset_index(drop=True, inplace=True)
        return file


class ArculusDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = "unlabeled"
        sample = torch.from_numpy(sample)
        sample = torch.unsqueeze(sample, 0)
        return sample, label


def __arculus_data_loader__(index: int, batch_size: int = 64):
    data = load_data_create_partitions(index=index)
    data.drop(["device_index", "index", "timestamp"], axis=1, inplace=True)
    data_columns = data.columns

    x = data[data_columns].values.astype(float)  # returns a numpy array
    x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
    x[np.isnan(x)] = 0.0
    data_normed = data.copy()
    data_normed[data_columns] = pd.DataFrame(x, columns=data_columns)

    base_transforms_list = [
        # transforms.ToPILImage(),
        # transforms.RandomRotation(90, fill=(0,)),
        # transforms.Pad(padding=4),
        # transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ]
    base_transform = transforms.Compose(base_transforms_list)
    dataset = ArculusDataset(data=data_normed[data_columns].to_numpy(),
                             transform=base_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_arculus_partition(index: int, *args, **kwargs):
    return __arculus_data_loader__(index)
