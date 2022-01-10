from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl

sys.path.insert(0, ".")
from src.training import train, test


class OneLayerAutoencoder(nn.Module):
    def __init__(self):
        super(OneLayerAutoencoder, self).__init__()
        compression = 0.7
        self.enc = nn.Linear(in_features=28*28, out_features=int(compression*28*28))
        self.dec = nn.Linear(in_features=int(compression*28*28), out_features=28*28)

    def forward(self, x):
        x = torch.relu(self.enc(x))
        x = torch.sigmoid(self.dec(x))
        return x


class MnistClient(fl.client.NumPyClient):

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, client_index: int):
        super(MnistClient, self).__init__()
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
        os_federated, _ = test(self.federated_detector, self.testloader)
        train(self.federated_detector, self.trainloader, epochs=1, verbose=False)
        train(self.ondevice_detector, self.trainloader, epochs=1, verbose=False)
        return self.get_parameters(), len(self.trainloader), {}

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
