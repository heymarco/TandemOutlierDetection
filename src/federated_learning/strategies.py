import sys
from ast import literal_eval
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from flwr.server.strategy import FedAdagrad, FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Weights, Scalar

from src.helper import save_outlier_scores


class TandemStrategy(FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_eval: float = 1.0,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = False,
            initial_parameters: Weights = None,
            exp_repetition: int = 1,
            to_csv: bool = False,
            num_rounds: int = 20,
    ) -> None:
        super(TandemStrategy, self).__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval,
                                             min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients,
                                             min_available_clients=min_available_clients,
                                             eval_fn=eval_fn, on_fit_config_fn=on_fit_config_fn,
                                             on_evaluate_config_fn=on_evaluate_config_fn,
                                             accept_failures=accept_failures, initial_parameters=initial_parameters)
        self.exp_repetition = exp_repetition
        self.to_csv = to_csv
        self.num_rounds = num_rounds

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        _ = super(TandemStrategy, self).aggregate_evaluate(rnd, results, failures)
        results = sorted(results, key=lambda tup: tup[0].cid)
        print(np.mean([tup[1].loss for tup in results]))
        print(rnd)
        if rnd == self.num_rounds:
            os_federated = [
                np.frombuffer(evaluate_res.metrics["os_federated"], dtype=float)
                for _, evaluate_res in results
            ]
            os_ondevice = [
                np.frombuffer(evaluate_res.metrics["os_ondevice"], dtype=float)
                for _, evaluate_res in results
            ]
            labels = [
                np.array(literal_eval(evaluate_res.metrics["labels"]), dtype=np.str)
                for _, evaluate_res in results
            ]
            client_indices = [
                int(evaluate_res.metrics["client_index"])
                for _, evaluate_res in results
            ]
            if self.to_csv:
                print("Trying to save dataframe")
                save_outlier_scores(client_indices=client_indices, os_federated=os_federated, os_ondevice=os_ondevice,
                                    labels=labels, exp_repetition=self.exp_repetition)
        return None, None


class TandemAdagrad(FedAdagrad):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_eval: float = 1.0,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = False,
            initial_parameters: Weights = None,
            exp_repetition: int = 1,
            to_csv: bool = False,
            num_rounds: int = 20,
    ) -> None:
        super(TandemStrategy, self).__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval,
                                             min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients,
                                             min_available_clients=min_available_clients,
                                             eval_fn=eval_fn, on_fit_config_fn=on_fit_config_fn,
                                             on_evaluate_config_fn=on_evaluate_config_fn,
                                             accept_failures=accept_failures, initial_parameters=initial_parameters)
        self.exp_repetition = exp_repetition
        self.to_csv = to_csv
        self.num_rounds = num_rounds

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            print("Could not get results")
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            print(failures)
            return None
        results = sorted(results, key=lambda tup: tup[0].cid)
        print(np.mean([tup[1].loss for tup in results]))
        if rnd == self.num_rounds:
            os_federated = [
                np.frombuffer(evaluate_res.metrics["os_federated"], dtype=float)
                for _, evaluate_res in results
            ]
            os_ondevice = [
                np.frombuffer(evaluate_res.metrics["os_ondevice"], dtype=float)
                for _, evaluate_res in results
            ]
            labels = [
                np.array(literal_eval(evaluate_res.metrics["labels"]), dtype=np.str)
                for _, evaluate_res in results
            ]
            client_indices = [
                int(evaluate_res.metrics["client_index"])
                for _, evaluate_res in results
            ]
            if self.to_csv:
                save_outlier_scores(client_indices=client_indices, os_federated=os_federated, os_ondevice=os_ondevice,
                                    labels=labels, exp_repetition=self.exp_repetition)
        return None
