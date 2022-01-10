from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl

from src.training import train, test


class Encoder(nn.Module):
    def __init__(self, batch_size, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True)

    def forward(self, x):
        x = x.reshape((self.batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.batch_size, self.embedding_dim))


class Decoder(nn.Module):

    def __init__(self, batch_size, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((self.batch_size, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True)

        self.decoder = nn.LSTM(self.latent_dim, self.input_dim, self.num_layers, batch_first=True)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        # It is way more general that way
        last_hidden = torch.squeeze(last_hidden)
        last_hidden = torch.unsqueeze(last_hidden, 1)
        encoded = last_hidden.repeat((1, input.size()[1], 1))
        # Decode
        y, _ = self.decoder(encoded)
        return torch.squeeze(y)


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


class IpekClient(fl.client.NumPyClient):

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, client_index: int):
        super(IpekClient, self).__init__()
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
