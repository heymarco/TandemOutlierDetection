import os
import logging
import torch
import torchvision.transforms as transforms
from sklearn import preprocessing
import pandas as pd
import numpy as np
from flwr.common.logger import log
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler


def get_min_max():
    data_dir = os.path.join("data", "n_baiot")
    data = None
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    for filename in os.listdir(data_dir):
        if (not filename.endswith(".csv")):
            continue
        if filename == "data_summary.csv" or filename == "device_info.csv" or filename == "features.csv":
            continue
        print(filename)
        if "benign" in filename:
            device_index, data_type, _ = filename.split(".")
        else:
            device_index, data_type, _, _ = filename.split(".")
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file)
        data = file if data is None else pd.concat([data, file])
    data = data.to_numpy()
    print("mean: {}".format(np.mean(data)))
    print("95th perc.: {}".format(np.percentile(data, 95)))
    print("minimum: {}".format(np.min(data)))
    print("maximum: {}".format(np.max(data)))


def load_data(index: int, contamination: float = 0.001) -> pd.DataFrame:
    log(logging.INFO, "Loading data for device {} ...".format(index))
    data_dir = os.path.join("data", "n_baiot")
    inlier_data = None
    outlier_data = None
    assert os.path.exists(data_dir), "The directory '{}' does not exist".format(data_dir)
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        if filename == "data_summary.csv" or filename == "device_info.csv" or filename == "features.csv":
            continue
        if not filename.startswith(str(index)):
            continue
        if "benign" in filename:
            device_index, data_type, _ = filename.split(".")
        else:
            device_index, data_type, _, _ = filename.split(".")
        path_to_file = os.path.join(data_dir, filename)
        file = pd.read_csv(path_to_file)
        data_type_col = [data_type for _ in range(len(file))]
        device_index_col = [device_index for _ in range(len(file))]
        file["data_type"] = data_type_col
        file["device_index"] = device_index_col

        if "benign" in filename:
            inlier_data = file if inlier_data is None else pd.concat([inlier_data, file])
        else:
            outlier_data = file.iloc[:2000] if outlier_data is None else pd.concat([outlier_data, file.iloc[:2000]])
    inlier_data.reset_index(drop=True, inplace=True)
    nobs = len(inlier_data)
    nout = int(nobs * contamination)
    outlier_data.reset_index(drop=True, inplace=True)
    final_dataframe = pd.concat([
        inlier_data,
        outlier_data.sample(nout)
    ]).sample(frac=1)
    log(logging.INFO, "Done loading data.")
    final_dataframe.reset_index(drop=True, inplace=True)
    return final_dataframe


class BaIoTDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform, labels):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.from_numpy(sample)
        sample = torch.unsqueeze(sample, 0)
        return sample, label


def __baiot_data_loader__(index: int, batch_size: int = 64):
    data = load_data(index=index)
    labels = data["data_type"]
    data.drop(["device_index", "data_type"], axis=1, inplace=True)
    data_columns = data.columns

    x = data[data_columns].values  # returns a numpy array
    x = MinMaxScaler(feature_range=[-1, 1]).fit_transform(x)
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
    dataset = BaIoTDataset(data=data_normed[data_columns].to_numpy(),
                           transform=base_transform, labels=labels)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_baiot_partition(index: int, *args, **kwargs):
    return __baiot_data_loader__(index+1)
