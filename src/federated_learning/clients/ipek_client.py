from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl

from src.training import train, test


class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super(LSTMAutoencoder, self).__init__()
        dim = 9
        hidden_dim = 4
        self.encoder = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x
        # _, (last_hidden, _) = self.encoder(x)
        # encoded = last_hidden.repeat(x.shape)
        # print(encoded)
        # y, _ = self.decoder(encoded)
        # return torch.squeeze(y)


class OneLayerAutoencoder(nn.Module):
    def __init__(self):
        super(OneLayerAutoencoder, self).__init__()
        self.enc = nn.Linear(in_features=1395, out_features=500)
        self.dec = nn.Linear(in_features=500, out_features=1395)

    def forward(self, x):
        x = torch.relu(self.enc(x))
        x = torch.tanh(self.dec(x))
        return x


class IpekClient(fl.client.NumPyClient):

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, client_index: int):
        super(IpekClient, self).__init__()
        self.federated_detector = LSTMAutoencoder()
        self.ondevice_detector = LSTMAutoencoder()
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
        train(self.federated_detector, self.trainloader, epochs=1, lr=learning_rate, verbose=self.client_index==0)
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
