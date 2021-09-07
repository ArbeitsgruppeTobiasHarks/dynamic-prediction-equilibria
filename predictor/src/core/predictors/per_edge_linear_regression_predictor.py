from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.piecewise_linear import PiecewiseLinear


class PerEdgeLinearRegressionPredictor(Predictor):

    def __init__(self, models: List[LinearRegression], past_timesteps: int, future_timesteps: int, network: Network):
        super().__init__(network)
        self._models = models
        self._past_timesteps, self._future_timesteps = past_timesteps, future_timesteps

    def type(self) -> str:
        return "Per Edge Linear Regression Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        raise NotImplementedError()

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) -> List[PiecewiseLinear]:
        times = [phi + t for t in range(0, 21, 1)]
        edges = self.network.graph.edges
        assert len(edges) == len(old_queues)
        past_times = [phi - t for t in range(-self._past_timesteps + 1, 1)]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
        for e in edges:
            inputs = [old_queues[ie.id](t) for t in past_times for ie in e.node_from.incoming_edges] + \
                     [old_queues[oe.id](t) for t in past_times for oe in e.node_to.outgoing_edges] + \
                     [old_queues[e.id](t) for t in past_times]
            prediction = self._models[e.id].predict([inputs])[0]
            prediction: List[float] = [old_queues[e.id](phi), *prediction]
            cap = self.network.capacity[e.id]
            new_values = [0.] * len(prediction)
            new_values[0] = prediction[0]
            for t in range(1, len(new_values)):
                new_values[t] = 2 * prediction[t] - new_values[t - 1]
                new_values[t] = max(new_values[t], new_values[t - 1] - cap, 0.)
            queues[e.id] = PiecewiseLinear(
                times,
                new_values,
                0., 0.
            ).simplify()

        return queues

    @staticmethod
    def from_models(network: Network, models_dir: str, past_timesteps: int, future_timesteps: int):
        models = []
        for e in network.graph.edges:
            model_path = os.path.join(models_dir, f"edge-{e.id}-model.pickle")
            assert os.path.exists(model_path)
            with open(model_path, "rb") as file:
                models.append(pickle.load(file))
        return PerEdgeLinearRegressionPredictor(models, past_timesteps, future_timesteps, network)
