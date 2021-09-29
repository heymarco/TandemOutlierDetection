import os
import torch
import torchvision.transforms as transforms
from sklearn import preprocessing
import pandas as pd


class MelbourneDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform, labels = False):
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


def __melbourne_data_loader__(batch_size: int = 64):
    data = pd.read_csv(os.path.join("data", "melbourne-sensor-data.csv"))
    data_columns = ["temp_max", "temp_min", "temp_avg",
                    "light_max", "light_min", "light_avg",
                    "humidity_max", "humidity_min", "humidity_avg"]

    x = data[data_columns].values.T  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_scaled = min_max_scaler.fit_transform(x.T)
    data_normed = data.copy()
    data_normed[data_columns] = pd.DataFrame(x_scaled, columns=data_columns)
    del x

    grouped_data_normed = data_normed.groupby(by="boardid")

    base_transforms_list = [
        # transforms.ToPILImage(),
        # transforms.RandomRotation(90, fill=(0,)),
        # transforms.Pad(padding=4),
        # transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ]
    base_transform = transforms.Compose(base_transforms_list)
    datasets = [MelbourneDataset(data=partition[1][data_columns].to_numpy(),
                                 transform=base_transform)
                for partition in grouped_data_normed]
    return [(torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True),
             torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)) for ds in datasets]


def load_melbourne_partition(index: int, *args, **kwargs):
    return __melbourne_data_loader__()[index]

