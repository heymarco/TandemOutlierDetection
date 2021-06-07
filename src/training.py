import numpy as np
import torch
from matplotlib import pyplot as plt

from src.helper import get_device


def train(net, loader, epochs, lr=0.01, momentum=0.9, verbose: bool = False):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # optimizer = torch.optim.Adam(net.parameters())
    for e in range(epochs):
        for b, (data, _) in enumerate(loader):
            data = data.to(get_device(), dtype=torch.float)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            if verbose and e % 10 == 0 and b == len(loader)-1:
                plt.clf()
                plt.plot(range(len(output[0])), output[0][:, 0].detach().numpy(), label="reconstruction")
                plt.plot(range(len(data[0])), data[0][:, 0].detach().numpy(), label="original")
                plt.legend()
                plt.show()


def test(net, loader):
    """Validate the network on the entire test set."""
    outlier_scores = []
    labels = []
    with torch.no_grad():
        for data, newlabels in loader:
            data = data.to(get_device(), dtype=torch.float)
            outputs = net(data)
            loss = [np.linalg.norm(out-original) for out, original in zip(outputs, data)]
            outlier_scores += loss
            labels += list(newlabels)
    return outlier_scores, labels