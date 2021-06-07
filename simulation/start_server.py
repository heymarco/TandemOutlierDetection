import argparse
import sys
from logging import WARNING
import flwr as fl
from flwr.common.logger import logger

sys.path.insert(0, ".")
from src.federated_learning.strategies import TandemStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_rounds", type=int, required=True, help="The number of communication rounds")
    parser.add_argument("-exp_index", type=int, default=0,
                        help="The current experiment repetition. May override old data")
    parser.add_argument("-to_csv", type=bool, default=False, required=True, help="Save result?")
    parser.add_argument("-type", type=str, choices=["melbourne", "baiot", "arculus", "ipek", "mnist", "synth"],
                        required=True,
                        help="The clients to start")
    args = parser.parse_args()
    logger.setLevel(WARNING)
    num_clients = 9
    if args.type == "arculus":
        num_clients = 19
    if args.type == "ipek":
        num_clients = 15
    if args.type == "mnist":
        num_clients = 8
    if args.type == "synth":
        num_clients = 30
    server_config = {
        "num_rounds": args.num_rounds
    }
    # Create strategy
    strategy = TandemStrategy(
        min_fit_clients=num_clients,
        min_eval_clients=num_clients,
        min_available_clients=num_clients,
        exp_repetition=args.exp_index,
        to_csv=args.to_csv
    )
    fl.server.start_server(config=server_config, strategy=strategy)
    print("exiting")
    sys.exit()
