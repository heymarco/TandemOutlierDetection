import argparse
import sys
import numpy as np

import flwr as fl

sys.path.insert(0, ".")

from src.federated_learning.clients.synthetic_client import SyntheticClient
from src.federated_learning.data.synthetic_data_loader import load_synthetic_partition

from src.federated_learning.clients.mnist_client import MnistClient
from src.federated_learning.data.mnist_data_loader import load_mnist_partition

from src.federated_learning.clients.ipek_client import IpekClient
from src.federated_learning.data.ipek_data_loader import load_ipek_partition

from src.federated_learning.clients.arculus_client import ArculusClient
from src.federated_learning.data.arculus_data_loader import load_arculus_partition

from src.federated_learning.clients.baiot_client import BaIoTClient
from src.federated_learning.data.baiot_data_loader import load_baiot_partition
from src.federated_learning.data.melbourne_data_loader import load_melbourne_partition
from src.federated_learning.clients.melbourne_client import MelbourneSensorDataClient


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, choices=["melbourne", "baiot", "arculus", "ipek", "mnist", "synth"],
                        required=True, help="The clients to start")
    parser.add_argument("-client_index", type=int, required=True, help="The index of clients")
    args = parser.parse_args()

    if args.type == "melbourne":
        train, test = load_melbourne_partition(args.client_index)
        client = MelbourneSensorDataClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "baiot":
        train, test = load_baiot_partition(args.client_index)
        client = BaIoTClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "arculus":
        train, test = load_arculus_partition(args.client_index)
        client = ArculusClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "ipek":
        train, test = load_ipek_partition(args.client_index)
        client = IpekClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "mnist":
        train, test = load_mnist_partition(args.client_index)
        client = MnistClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "synth":
        train, test = load_synthetic_partition(args.client_index)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)


    fl.client.start_numpy_client("[::]:8080", client=client)
    sys.exit()
