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
    parser.add_argument("-num_reps", type=int,
                        help="The current experiment repetition. May override old data")
    parser.add_argument("-num_clients", type=int, required=False,
                        help="The number of clients")
    parser.add_argument("-to_csv", type=bool, default=False, required=True, help="Save result?")
    parser.add_argument("-type", type=str, choices=["powertool", "local/global", "partition_outlier"],
                        required=True,
                        help="The clients to start")
    args = parser.parse_args()
    logger.setLevel(WARNING)
    num_clients = args.num_clients
    server_config = {
        "num_rounds": args.num_rounds
    }
    # Create strategy
    strategy = TandemStrategy(
        min_fit_clients=num_clients,
        min_eval_clients=num_clients,
        min_available_clients=num_clients,
        exp_repetition=args.exp_index,
        to_csv=args.to_csv,
        num_rounds=args.num_rounds
    )
    fl.server.start_server(config=server_config, strategy=strategy)
    sys.exit()
