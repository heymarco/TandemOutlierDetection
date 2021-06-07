from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl

from src.training import train, test


class OneLayerAutoencoder(nn.Module):
    def __init__(self):
        super(OneLayerAutoencoder, self).__init__()
        self.enc = nn.Linear(in_features=20, out_features=8)
        self.dec = nn.Linear(in_features=8, out_features=20)

    def forward(self, x):
        x = self.enc(x)
        x = torch.relu(x)
        x = self.dec(x)
        x = torch.tanh(x)
        return x


class SyntheticClient(fl.client.NumPyClient):

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, client_index: int):
        super(SyntheticClient, self).__init__()
        self.federated_detector = OneLayerAutoencoder()
        self.ondevice_detector = OneLayerAutoencoder()
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_index = client_index

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.federated_detector.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.federated_detector.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.federated_detector.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr_local = 0.01
        lr_federated = lr_local
        train(self.federated_detector, self.trainloader, epochs=1, lr=lr_federated)
        train(self.ondevice_detector, self.trainloader, epochs=1, lr=lr_local)
        return self.get_parameters(), 1000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        os_federated, _ = test(self.federated_detector, self.testloader)
        os_ondevice, labels = test(self.ondevice_detector, self.testloader)
        return float(np.mean(os_federated)), len(self.testloader), {
            "os_federated": np.array(os_federated, dtype=float).tobytes(),
            "os_ondevice": np.array(os_ondevice, dtype=float).tobytes(),
            "labels": str(labels),
            "client_index": self.client_index
        }
