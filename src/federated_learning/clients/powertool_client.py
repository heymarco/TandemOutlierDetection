from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl

from src.training import train, test


class OneLayerAutoencoder(nn.Module):
    def __init__(self):
        super(OneLayerAutoencoder, self).__init__()
        in_features = 1665
        enc_factor = 0.7
        enc_features = int(enc_factor * in_features)
        self.enc = nn.Linear(in_features=in_features, out_features=enc_features)
        self.dec = nn.Linear(in_features=enc_features, out_features=in_features)
        self.do = nn.Dropout(p=0.0, inplace=False)

    def forward(self, x):
        x = torch.relu(self.enc(x))
        x = self.do(x)
        x = self.dec(x)
        return x


class PowertoolClient(fl.client.NumPyClient):

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, client_index: int):
        super(PowertoolClient, self).__init__()
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
        learning_rate = 0.01
        train(self.federated_detector, self.trainloader, epochs=1, lr=learning_rate, verbose=False)
        train(self.ondevice_detector, self.trainloader, epochs=1, lr=learning_rate, verbose=False)
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
