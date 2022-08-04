from __future__ import annotations

from typing import List, Optional

import numpy as np
from core.dynamic_flow import DynamicFlow
import tensorflow as tf

from core.network import Network
from core.predictor import ComputeMode, Predictor
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

    def compute_mode(self) -> ComputeMode:
        return ComputeMode.DYNAMIC

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time +
                 t for t in range(0, self._future_timesteps + 1)]
        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        assert len(edges) == len(flow.queues)
        input_times = [prediction_time + t *
                       self._prediction_interval for t in range(-self._past_timesteps+1, 1)]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)

        edge_loads = flow.get_edge_loads()

        data = np.array([
            [
                [queue(t) for t in input_times]
                for e_id, queue in enumerate(flow.queues)
                if self._test_mask[e_id]
            ],
            [
                [load(t) for t in input_times]
                for e_id, load in enumerate(edge_loads)
                if self._test_mask[e_id]
            ]
        ])

        future_queues_raw = self._model.predict(np.array([[prediction_time, *(data.flatten())]]), verbose=0)[0]
        future_queues_raw = np.maximum(
            future_queues_raw, np.zeros_like(future_queues_raw))
        for e_id, old_queue in enumerate(flow.queues):
            if not self._test_mask[e_id]:
                queues[e_id] = zero_fct
                continue
            masked_id = np.count_nonzero(self._test_mask[:e_id])

            new_values = [
                old_queue(prediction_time),
                *future_queues_raw[
                    masked_id * self._future_timesteps: (masked_id + 1) * self._future_timesteps]
            ]

            for i in range(1, len(new_values)):
                new_values[i] = max(
                    new_values[i], new_values[i-1] - self._prediction_interval * self._network.capacity[e_id])

            queues[e_id] = PiecewiseLinear(times, new_values, 0., 0.)

        return queues

    def batch_predict(self, prediction_times: List[float], flow: DynamicFlow) -> List[List[PiecewiseLinear]]:
        edge_loads = flow.get_edge_loads()

        evaluated_queues = {}
        evaluated_loads = {}
        number_masked_ids = np.count_nonzero(self._test_mask)

        for e_id, old_queue in enumerate(flow.queues):
            if not self._test_mask[e_id]:
                continue
            masked_id = np.count_nonzero(self._test_mask[:e_id])
            evaluated_queues[masked_id] = {}
            evaluated_loads[masked_id] = {}

            for prediction_time in prediction_times:
                for t in range(-self._past_timesteps+1, 1):
                    time = prediction_time + t * self._prediction_interval
                    if time not in evaluated_queues[masked_id]:
                        evaluated_queues[masked_id][time] = old_queue(time)
                        evaluated_loads[masked_id][time] = edge_loads[e_id](
                            time)

        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([0.], [0.], 0., 0.)
        assert len(edges) == len(flow.queues)

        result_predictions = []

        raw_predictions_input = np.array([
            [
                prediction_time,
                *[
                    evaluated_queues[masked_id][prediction_time +
                                                t*self._prediction_interval]
                    for masked_id in range(number_masked_ids)
                    for t in range(-self._past_timesteps+1, 1)
                ],
                *[
                    evaluated_loads[masked_id][prediction_time +
                                               t*self._prediction_interval]
                    for masked_id in range(number_masked_ids)
                    for t in range(-self._past_timesteps+1, 1)
                ]
            ]
            for prediction_time in prediction_times
        ])
        future_queues_raw = self._model.predict(raw_predictions_input, verbose=0)
        future_queues_raw = np.maximum(future_queues_raw, np.zeros_like(future_queues_raw))

        for prediction_ind, prediction_time in enumerate(prediction_times):
            queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)

            masked_id = -1
            times = [prediction_time + t *
                     self._prediction_interval for t in range(0, self._future_timesteps + 1)]
            for e_id, old_queue in enumerate(flow.queues):
                if not self._test_mask[e_id]:
                    queues[e_id] = zero_fct
                    continue
                masked_id += 1

                new_values = [
                    evaluated_queues[masked_id][prediction_time],
                    *future_queues_raw[prediction_ind,
                        masked_id * self._future_timesteps: (masked_id + 1) * self._future_timesteps]
                ]

                for i in range(1, len(new_values)):
                    new_values[i] = max(
                        new_values[i], new_values[i-1] - self._prediction_interval * self._network.capacity[e_id])

                queues[e_id] = PiecewiseLinear(times, new_values, 0., 0.)

            result_predictions.append(queues)
        return result_predictions

    @staticmethod
    def from_model(network: Network, model_path: str, test_mask, past_timesteps: int, future_timesteps: int, prediction_interval: float):
        model = tf.keras.models.load_model(model_path)
        return TFFullNetPredictor(model, test_mask, network, past_timesteps, future_timesteps, prediction_interval)
