import argparse
import sys
import numpy as np

import flwr as fl

sys.path.insert(0, ".")

from src.federated_learning.clients.synthetic_client import SyntheticClient
from src.federated_learning.data.synthetic_data_loader import load_synthetic_partition

from src.federated_learning.clients.ipek_client import IpekClient
from src.federated_learning.data.ipek_data_loader import load_ipek_partition


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str,
                        choices=["powertool", "local/global", "partition_outlier"],
                        required=True, help="The clients to start")
    parser.add_argument("-client_index", type=int, required=True, help="The index of clients")
    parser.add_argument("-exp_index", type=int, required=True, help="The experiment index")
    parser.add_argument("-num_reps", type=int, required=True, help="The number of experiment repetitions")
    args = parser.parse_args()

    if args.type == "powertool":
        train, test = load_ipek_partition(args.client_index)
        client = IpekClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "local/global":
        train, test = load_synthetic_partition(index=args.client_index, experiment=args.type,
                                               nobs=1000, **args.__dict__)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)
    if args.type == "partition_outlier":
        train, test = load_synthetic_partition(index=args.client_index, experiment=args.type,
                                               nobs=1000, **args.__dict__)
        client = SyntheticClient(trainloader=train, testloader=test, client_index=args.client_index)

    fl.client.start_numpy_client("[::]:8080", client=client)
