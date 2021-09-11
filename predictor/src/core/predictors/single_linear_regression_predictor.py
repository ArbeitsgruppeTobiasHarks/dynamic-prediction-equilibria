from __future__ import annotations

import os
import pickle
import re
import string
from typing import List, Optional, Dict, Callable

import numpy as np
from sklearn.linear_model import LinearRegression

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.piecewise_linear import PiecewiseLinear


class SingleLinearRegressionPredictor(Predictor):

    def __init__(self, predict: Callable[[List[PiecewiseLinear], float], List[PiecewiseLinear]], network: Network):
        super().__init__(network)
        self._predict = predict

    def type(self) -> str:
        return "Single Linear Regression Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        raise NotImplementedError()

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) -> List[PiecewiseLinear]:
        return self._predict(old_queues, phi)

    @staticmethod
    def from_weka_models(network: Network, models_folder: str):
        coefficients = {}
        past_timesteps, future_timesteps = 10, 10
        for i in range(1, future_timesteps + 1):
            d = {}
            with open(os.path.join(models_folder, f"e{i}.txt")) as file:
                content = file.read()
            start_ind = content.find(f"e[{i}] =") + len(f"e[{i}] =\n\n")
            if start_ind < 0:
                raise ValueError()
            matches = re.findall(r"(-?[0-9]*(?:\.[0-9]*)?) *\* *([ei\-0-9\[\]]*) \+\n", content)
            for match in matches:
                d[match[1]] = float(match[0])
            constant_match = re.search(r"-?[0-9]*(?:\.[0-9]*)? *\* *[ei\-0-9\[\]]* \+\n *(-?[0-9]*(?:\.[0-9]*)?)\n",
                                       content)
            if constant_match is None:
                raise ValueError()
            d["c"] = float(constant_match.groups()[0])
            coefficients[f"e[{i}]"] = d

        def predict(old_queues: List[PiecewiseLinear], phi: float):
            times = [phi + t for t in range(0, future_timesteps)]
            step_size = 1.0
            edges = network.graph.edges
            assert len(edges) == len(old_queues)
            queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
            for e_id, old_queue in enumerate(old_queues):
                inputs: Dict[string, float] = dict()
                for i in range(past_timesteps + 1):
                    inputs[f"e[{-i}]"] = max(0., old_queue(phi - i * step_size))
                for k, e in enumerate(edges[e_id].node_from.outgoing_edges):
                    for i in range(11):
                        inputs[f"i{k}[{-i}]"] = max(0., old_queues[e.id](phi - i * step_size))

                new_values = [inputs["e[0]"]] + [0.] * future_timesteps
                for i in range(1, 11):
                    d = coefficients[f"e[{i}]"]
                    new_values[i] = sum(
                        inputs[key] * value for (key, value) in d.items() if key in inputs
                    ) + d["c"]
                queues[e_id] = PiecewiseLinear(
                    times,
                    new_values,
                    0., 0.
                )
            return queues

        return SingleLinearRegressionPredictor(predict, network)

    @staticmethod
    def from_scikit_model(network: Network, model_path: str, past_timesteps: int, future_timesteps: int):
        with open(model_path, "rb") as file:
            model: LinearRegression = pickle.load(file)

        def predict(old_queues: List[PiecewiseLinear], phi: float):
            queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
            for e in network.graph.edges:
                inputs = [0.] * ((5 + 1) * past_timesteps)
                for k, ie in enumerate(e.node_from.incoming_edges):
                    inputs[k * past_timesteps: (k + 1) * past_timesteps] = [
                        old_queues[ie.id](phi + t) for t in range(-past_timesteps + 1, 1)
                    ]
                inputs[5 * past_timesteps: (5 + 1) * past_timesteps] = [
                    old_queues[e.id](phi + t) for t in range(-past_timesteps + 1, 1)
                ]
                if any(v != 0. for v in inputs):
                    prediction = model.predict([inputs])[0]
                    new_values = [old_queues[e.id](phi), *prediction]
                    queues[e.id] = PiecewiseLinear(
                        [phi + t for t in range(future_timesteps + 1)],
                        new_values,
                        0., 0.
                    )
                else:
                    queues[e.id] = PiecewiseLinear([phi], [0.], 0., 0.)
            return queues

        return SingleLinearRegressionPredictor(predict, network)
