from __future__ import annotations

import os
import re
import string
from typing import List, Optional, Dict

import numpy as np

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.piecewise_linear import PiecewiseLinear


class ExpandedLinearRegressionPredictor(Predictor):

    def __init__(self, coefficients: Dict[string, Dict[string, float]], network: Network):
        super().__init__(network)
        self._coefficients = coefficients

    def type(self) -> str:
        return "Expanded Linear Regression Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        raise NotImplementedError()

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) -> List[PiecewiseLinear]:
        times = [phi + t for t in range(0, 21, 2)]
        step_size = 2.0
        edges = self.network.graph.edges
        assert len(edges) == len(old_queues)
        queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
        for e_id, old_queue in enumerate(old_queues):
            inputs: Dict[string, float] = dict()
            for i in range(11):
                inputs[f"e[{-i}]"] = max(0., old_queue(phi - i * step_size))
            for k, e in enumerate(edges[e_id].node_from.outgoing_edges):
                for i in range(11):
                    inputs[f"i{k}[{-i}]"] = max(0., old_queues[e.id](phi - i * step_size))

            new_values = [inputs["e[0]"]] + [0.] * 10
            for i in range(1, 11):
                d = self._coefficients[f"e[{i}]"]
                new_values[i] = sum(
                    inputs[key] * value for (key, value) in d.items() if key in inputs
                ) + d["c"]
            queues[e_id] = PiecewiseLinear(
                times,
                new_values,
                0., 0.
            )

        return queues

    @staticmethod
    def from_models(network: Network, models_folder: str):
        coefficients = {}
        for i in range(1, 11):
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
        return ExpandedLinearRegressionPredictor(coefficients, network)
