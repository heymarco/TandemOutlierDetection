import numpy as np
import torch

from src.helper import get_device


def train(net, loader, epochs, lr=0.01, momentum=0.9, verbose: bool = False):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    for e in range(epochs):
        for b, (data, _) in enumerate(loader):
            data = data.to(get_device(), dtype=torch.float)
            optimizer.zero_grad()
            output = net(data)
            assert output.size() == data.size()
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()


def test(net, loader, verbose=False):
    """Validate the network on the entire test set."""
    outlier_scores = []
    labels = []
    with torch.no_grad():
        for bidx, (data, newlabels) in enumerate(loader):
            data = data.to(get_device(), dtype=torch.float)
            outputs = net(data)
            loss = [np.linalg.norm(out-original) for out, original in zip(outputs, data)]
            outlier_scores += loss
            labels += list(newlabels)
    return outlier_scores, labels