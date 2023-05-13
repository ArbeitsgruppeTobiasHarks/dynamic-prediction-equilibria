from __future__ import annotations
import os

from typing import Dict, List, Optional

import numpy as np
from core.dynamic_flow import DynamicFlow
import tensorflow as tf

from core.network import Network
from core.predictor import Predictor
from ml.neighboring_edges import get_neighboring_edges_mask_undirected
from utilities.piecewise_linear import PiecewiseLinear


class TFNeighborhoodPredictor(Predictor):

    def __init__(self, models: Dict[int, tf.keras.Sequential], input_mask: np.ndarray, output_mask: np.ndarray,
                 network: Network, past_timesteps: int, future_timesteps: int, prediction_interval: float,
                 max_distance: int):
        super().__init__(network)
        self._models = models
        self._input_mask = input_mask
        self._output_mask = output_mask
        self._network = network
        self._past_timesteps = past_timesteps
        self._future_timesteps = future_timesteps
        self._prediction_interval = prediction_interval
        self._edge_input_masks = [
            get_neighboring_edges_mask_undirected(
                edge, network, max_distance) * input_mask
            for edge in network.graph.edges
        ]

    def type(self) -> str:
        return "Neighborhood Neural Net Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time +
                 t for t in range(0, self._future_timesteps + 1)]
        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        assert len(edges) == len(flow.queues)
        input_times = [prediction_time + t *
                       self._prediction_interval for t in range(-self._past_timesteps + 1, 1)]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)

        edge_loads = flow.get_edge_loads()

        data = np.array([
            [
                queue.eval_sorted_array(input_times)
                if self._input_mask[e_id]
                else [0 for _ in input_times]
                for e_id, queue in enumerate(flow.queues)
            ],
            [
                load.eval_sorted_array(input_times)
                if self._input_mask[e_id]
                else [0 for _ in input_times]
                for e_id, load in enumerate(edge_loads)
            ]
        ])

        for e_id, old_queue in enumerate(flow.queues):
            if not self._output_mask[e_id]:
                queues[e_id] = zero_fct
                continue

            future_queues_raw = self._models[e_id].predict(
                np.array([[prediction_time, *(data[:, self._edge_input_masks[e_id], :].flatten())]]), verbose=0)[0]
            future_queues_raw = np.maximum(
                future_queues_raw, np.zeros_like(future_queues_raw))

            new_values = [
                old_queue(prediction_time),
                *future_queues_raw
            ]

            for i in range(1, len(new_values)):
                new_values[i] = max(
                    new_values[i], new_values[i - 1] - self._prediction_interval * self._network.capacity[e_id])

            queues[e_id] = PiecewiseLinear(times, new_values, 0., 0.)

        return queues

    def batch_predict(self, prediction_times: List[float], flow: DynamicFlow) -> List[List[PiecewiseLinear]]:
        edge_loads = flow.get_edge_loads()

        evaluated_queues = {}
        evaluated_loads = {}
        number_input_masked_ids = np.count_nonzero(self._input_mask)

        for e_id, queue in enumerate(flow.queues):
            if not self._input_mask[e_id]:
                continue
            masked_id = np.count_nonzero(self._input_mask[:e_id])
            evaluated_queues[masked_id] = {}
            evaluated_loads[masked_id] = {}

            for prediction_time in prediction_times:
                for t in range(-self._past_timesteps + 1, 1):
                    time = prediction_time + t * self._prediction_interval
                    if time not in evaluated_queues[masked_id]:
                        evaluated_queues[masked_id][time] = queue(time)
                        evaluated_loads[masked_id][time] = edge_loads[e_id](
                            time)

        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([0.], [0.], 0., 0.)
        assert len(edges) == len(flow.queues)

        result_predictions = [
            [] for _ in prediction_times
        ]

        for e_id, queue in enumerate(flow.queues):
            if not self._output_mask[e_id]:
                for prediction_ind, _ in enumerate(prediction_times):
                    result_predictions[prediction_ind].append(zero_fct)
                continue

            raw_predictions_input = np.array([
                [
                    prediction_time,
                    *[
                        evaluated_queues[masked_id][prediction_time +
                                                    t * self._prediction_interval]
                        for masked_id in range(number_input_masked_ids)
                        if self._edge_input_masks[e_id][self._input_mask][masked_id]
                        for t in range(-self._past_timesteps + 1, 1)
                    ],
                    *[
                        evaluated_loads[masked_id][prediction_time +
                                                   t * self._prediction_interval]
                        for masked_id in range(number_input_masked_ids)
                        if self._edge_input_masks[e_id][self._input_mask][masked_id]
                        for t in range(-self._past_timesteps + 1, 1)
                    ]
                ]
                for prediction_time in prediction_times
            ])

            future_queues_raw = self._models[e_id].predict(
                raw_predictions_input, verbose=0)
            future_queues_raw = np.maximum(
                future_queues_raw, np.zeros_like(future_queues_raw))

            masked_id = np.count_nonzero(self._input_mask[:e_id])
            for prediction_ind, prediction_time in enumerate(prediction_times):
                times = [prediction_time + t *
                         self._prediction_interval for t in range(0, self._future_timesteps + 1)]
                new_values = [
                    evaluated_queues[masked_id][prediction_time],
                    *future_queues_raw[prediction_ind, :]
                ]

                for i in range(1, len(new_values)):
                    new_values[i] = max(
                        new_values[i], new_values[i - 1] - self._prediction_interval * self._network.capacity[e_id])
                result_predictions[prediction_ind].append(PiecewiseLinear(times, new_values, 0., 0.))

        return result_predictions

    @staticmethod
    def from_models(network: Network, models_path: str, input_mask: np.ndarray, output_mask: np.ndarray,
                    past_timesteps: int, future_timesteps: int, prediction_interval: float, max_distance: int):
        models = {}
        for edge in network.graph.edges:
            if output_mask[edge.id] == 1:
                models[edge.id] = tf.keras.models.load_model(
                    os.path.join(models_path, str(edge.id)))
        return TFNeighborhoodPredictor(models, input_mask, output_mask, network, past_timesteps, future_timesteps,
                                       prediction_interval, max_distance)
