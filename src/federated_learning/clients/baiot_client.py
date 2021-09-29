import sys
from collections import OrderedDict

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
        dim = 115
        enc_size = int(0.4*dim)
        self.drop_layer = nn.Dropout(p=0.1)
        self.enc = nn.Linear(in_features=dim, out_features=enc_size)
        self.dec = nn.Linear(in_features=enc_size, out_features=dim)

    def forward(self, x):
        x = self.drop_layer(x)
        x = torch.relu(self.enc(x))
        x = torch.tanh(self.dec(x))
        return x


class BaIoTClient(fl.client.NumPyClient):

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, client_index: int):
        super(BaIoTClient, self).__init__()
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
        train(self.ondevice_detector, self.trainloader, epochs=1)
        train(self.federated_detector, self.trainloader, epochs=1)
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
