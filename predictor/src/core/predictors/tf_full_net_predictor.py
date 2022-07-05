from __future__ import annotations

from typing import List, Optional

import numpy as np
from core.dynamic_flow import DynamicFlow
import tensorflow as tf

from core.network import Network
from core.predictor import Predictor
from utilities.piecewise_linear import PiecewiseLinear


class TFFullNetPredictor(Predictor):

    def __init__(self, model: tf.keras.Sequential, test_mask: np.ndarray, network: Network, past_timesteps: int, future_timesteps: int, prediction_interval: float):
        super().__init__(network)
        self._model = model
        self._test_mask = test_mask
        self._network = network
        self._past_timesteps = past_timesteps
        self._future_timesteps = future_timesteps
        self._prediction_interval = prediction_interval

    def type(self) -> str:
        return "Full Net Linear Regression Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time +
                 t for t in range(0, self._future_timesteps + 1)]
        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        assert len(edges) == len(flow.queues)
        input_times = [prediction_time + t*self._prediction_interval for t in range(-self._past_timesteps+1, 1)]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)

        edge_loads = flow.get_edge_loads()

        phi = flow.phi
        data = np.asarray([
            [
                [queue(t) for t in input_times]
                for queue in flow.queues
            ],
            [
                [load(t) for t in input_times]
                for load in edge_loads
            ]
        ])

        # past_data = np.reshape(data[:, :, sample_id: phi_ind], newshape=(len(self._network.graph.edges), 2*self._past_timesteps))
        past_data = np.reshape(data, newshape=(len(self._network.graph.edges), 2*self._past_timesteps))

        future_queues_raw = self._model.predict([[phi, *(past_data[self._test_mask].flatten())]], verbose=0)[0]
        future_queues_raw = np.maximum(
            future_queues_raw, np.zeros_like(future_queues_raw))
        for e_id, old_queue in enumerate(flow.queues):
            if not self._test_mask[e_id]:
                queues[e_id] = zero_fct
                continue
            masked_id = np.count_nonzero(self._test_mask[:e_id])

            new_values = [
                old_queue(phi),
                *future_queues_raw[
                    masked_id * self._future_timesteps: (masked_id + 1) * self._future_timesteps]
            ]

            for i in range(1, len(new_values)):
                new_values[i] = max(
                    new_values[i], new_values[i-1] - self._prediction_interval * self._network.capacity[e_id])

            queues[e_id] = PiecewiseLinear(times, new_values, 0., 0.)

        return queues

    @staticmethod
    def from_model(network: Network, model_path: str, test_mask, past_timesteps: int, future_timesteps: int, prediction_interval: float):
        model = tf.keras.models.load_model(model_path)
        return TFFullNetPredictor(model, test_mask, network, past_timesteps, future_timesteps, prediction_interval)
