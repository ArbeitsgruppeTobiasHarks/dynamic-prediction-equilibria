from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from core.dynamic_flow import DynamicFlow

from core.network import Network
from core.predictor import ComputeMode, Predictor
from utilities.piecewise_linear import PiecewiseLinear


class PerEdgeLinearRegressionPredictor(Predictor):

    def __init__(self, models: List[LinearRegression], past_timesteps: int, future_timesteps: int, network: Network, average: bool):
        super().__init__(network)
        self._models = models
        self._past_timesteps, self._future_timesteps = past_timesteps, future_timesteps
        self._average = average

    def type(self) -> str:
        return "Per Edge Linear Regression Predictor"

    def compute_mode(self) -> ComputeMode:
        return ComputeMode.DYNAMIC

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time + t for t in range(0, self._future_timesteps + 1, 1)]
        edges = self.network.graph.edges
        assert len(edges) == len(flow.queues)
        past_times = [prediction_time - t for t in range(-self._past_timesteps + 1, 1)]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)
        for e in edges:
            inputs = [flow.queues[ie.id](t) for t in past_times for ie in e.node_from.incoming_edges] + \
                     [flow.queues[oe.id](t) for t in past_times for oe in e.node_to.outgoing_edges] + \
                     [flow.queues[e.id](t) for t in past_times]
            prediction = self._models[e.id].predict([inputs])[0]
            prediction: List[float] = [flow.queues[e.id](prediction_time), *prediction]
            cap = self.network.capacity[e.id]
            new_values = [0.] * len(prediction)
            new_values[0] = prediction[0]
            for t in range(1, len(new_values)):
                new_values[t] = 2 * prediction[t] - new_values[t - 1] if self._average else prediction[t]
                new_values[t] = max(new_values[t], new_values[t - 1] - cap, 0.)
            queues[e.id] = PiecewiseLinear(
                times,
                new_values,
                0., 0.
            ).simplify()

        return queues

    @staticmethod
    def from_models(network: Network, models_dir: str, past_timesteps: int, future_timesteps: int, average: bool):
        models = []
        for e in network.graph.edges:
            model_path = os.path.join(models_dir, f"edge-{e.id}-model.pickle")
            assert os.path.exists(model_path)
            with open(model_path, "rb") as file:
                models.append(pickle.load(file))
        return PerEdgeLinearRegressionPredictor(models, past_timesteps, future_timesteps, network, average)
