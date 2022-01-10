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
    parser.add_argument("-type", type=str,
                        choices=["melbourne", "baiot", "arculus", "ipek", "mnist", "synth",
                                 "synth_vary_clients", "local/global", "partition_outlier"],
                        required=True, help="The clients to start")
    parser.add_argument("-client_index", type=int, required=True, help="The index of clients")
    parser.add_argument("-exp_index", type=int, required=True, help="The experiment index")
    parser.add_argument("-num_reps", type=int, required=True, help="The number of experiment repetitions")
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
    if args.type == "local/global":
        train, test = load_synthetic_partition(index=args.client_index, experiment=args.type,
                                               nobs=1000, **args.__dict__)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "partition_outlier":
        train, test = load_synthetic_partition(index=args.client_index, experiment=args.type,
                                               nobs=1000, **args.__dict__)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "synth_vary_clients":
        experiment = "partition outlier"
        num_reps = args.num_reps
        current_index = args.exp_index
        repetitions_with_same_params = 10
        steps = 10
        counter_for_params = int(current_index / repetitions_with_same_params)
        affected_dims = 0.6
        variance = [0.0, 0.2, 0.4, 0.6][int(counter_for_params % 4)]
        print("Variance = {}".format(variance))
        shift_amount = 0.6
        nobs = 1000
        is_po = args.client_index == 0
        if args.client_index == 0:
            variance = 0.0
        if args.client_index > 0:
            shift_amount = 0
        print("Round {}, Affected dims = {}, Number of observations = {}, Shift amount = {}".format(
            current_index, affected_dims, nobs, shift_amount
        ))
        train, test = load_synthetic_partition(args.client_index, experiment=experiment,
                                               rep=args.exp_index, fraction=affected_dims, nobs=nobs,
                                               dims=20, shift=shift_amount, is_po=is_po, inter_client_variance=variance)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "synth":
        experiment = "partition outlier"
        vary = "divergence"  # nobs, affected_dims
        num_reps = args.num_reps
        current_index = args.exp_index
        repetitions_with_same_params = 10
        steps = 10
        counter_for_params = int(current_index / repetitions_with_same_params)
        affected_dims = 1.0
        variance = 0.0
        if vary == "n_po":
            nobs = 1000
            shift_amount = 0.4
            affected_dims = [0.1, 0.3, 0.6, 1.0][int(counter_for_params % 4)]
            num_po = [0, 1, 3, 6, 10, 20, 25, 30][int(counter_for_params / 4)]
            is_po = args.client_index < num_po
        elif vary == "nobs":
            nobs = [100, 300, 1000, 3000][int(counter_for_params / steps)]
            shift_amount = (counter_for_params % steps) / (steps - 1)
            is_po = args.client_index == 0
            if args.client_index > 0:
                shift_amount = 0
        elif vary == "affected_dims":
            affected_dims = [0.1, 0.3, 0.6, 1.0][int(counter_for_params / steps)]
            nobs = 1000
            shift_amount = (counter_for_params % steps) / (steps - 1)
            is_po = args.client_index == 0
            if args.client_index > 0:
                shift_amount = 0
        elif vary == "divergence":
            affected_dims = [0.1, 0.3, 0.6, 1.0][int(counter_for_params / steps)]
            nobs = 1000
            shift_amount = 0.6
            variance = (counter_for_params % steps) / (steps - 1)
            is_po = args.client_index == 0
            if args.client_index == 0:
                variance = 0.0
            if args.client_index > 0:
                shift_amount = 0
        print("Round {}, Affected dims = {}, Number of observations = {}, Shift amount = {}".format(
            current_index, affected_dims, nobs, shift_amount
        ))
        train, test = load_synthetic_partition(args.client_index, experiment=experiment,
                                               rep=args.exp_index, fraction=affected_dims, nobs=nobs,
                                               dims=20, shift=shift_amount, is_po=is_po, inter_client_variance=variance)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)

    fl.client.start_numpy_client("[::]:8080", client=client)
