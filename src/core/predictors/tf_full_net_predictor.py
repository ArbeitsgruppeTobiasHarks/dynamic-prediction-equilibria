from __future__ import annotations

from typing import List, Optional

import numpy as np
from core.dynamic_flow import DynamicFlow
import tensorflow as tf

from core.network import Network
from core.predictor import Predictor
from utilities.piecewise_linear import PiecewiseLinear


class TFFullNetPredictor(Predictor):
    def __init__(
        self,
        model: tf.keras.Sequential,
        input_mask: np.ndarray,
        output_mask: np.ndarray,
        network: Network,
        past_timesteps: int,
        future_timesteps: int,
        prediction_interval: float,
    ):
        super().__init__(network)
        self._model = model
        self._input_mask = input_mask
        self._output_mask = output_mask
        self._network = network
        self._past_timesteps = past_timesteps
        self._future_timesteps = future_timesteps
        self._prediction_interval = prediction_interval

    def type(self) -> str:
        return "Full Net Neural Net Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(
        self, prediction_time: float, flow: DynamicFlow
    ) -> List[PiecewiseLinear]:
        times = [
            prediction_time + t * self._prediction_interval
            for t in range(0, self._future_timesteps + 1)
        ]
        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([prediction_time], [0.0], 0.0, 0.0)
        assert len(edges) == len(flow.queues)
        input_times = [
            prediction_time + t * self._prediction_interval
            for t in range(-self._past_timesteps + 1, 1)
        ]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)

        edge_loads = flow.get_edge_loads()

        data = np.array(
            [
                [
                    queue.eval_sorted_array(input_times)
                    for e_id, queue in enumerate(flow.queues)
                    if self._input_mask[e_id]
                ],
                [
                    load.eval_sorted_array(input_times)
                    for e_id, load in enumerate(edge_loads)
                    if self._input_mask[e_id]
                ],
            ]
        )

        future_queues_raw = self._model.predict(
            np.array([[prediction_time, *(data.flatten())]]), verbose=0
        )[0]
        future_queues_raw = np.maximum(
            future_queues_raw, np.zeros_like(future_queues_raw)
        )
        for e_id, old_queue in enumerate(flow.queues):
            if not self._output_mask[e_id]:
                queues[e_id] = zero_fct
                continue
            masked_id = np.count_nonzero(self._output_mask[:e_id])

            new_values = [
                old_queue(prediction_time),
                *future_queues_raw[
                    masked_id
                    * self._future_timesteps : (masked_id + 1)
                    * self._future_timesteps
                ],
            ]

            for i in range(1, len(new_values)):
                new_values[i] = max(
                    new_values[i],
                    new_values[i - 1]
                    - self._prediction_interval * self._network.capacity[e_id],
                )

            queues[e_id] = PiecewiseLinear(times, new_values, 0.0, 0.0)

        return queues

    def batch_predict(
        self, prediction_times: List[float], flow: DynamicFlow
    ) -> List[List[PiecewiseLinear]]:
        edge_loads = flow.get_edge_loads()

        evaluated_queues = {}
        evaluated_loads = {}

        for e_id, queue in enumerate(flow.queues):
            if not self._input_mask[e_id]:
                continue
            evaluated_queues[e_id] = {}
            evaluated_loads[e_id] = {}

            evaluation_times = list(
                set(
                    prediction_time + t * self._prediction_interval
                    for prediction_time in prediction_times
                    for t in range(-self._past_timesteps + 1, 1)
                )
            )
            evaluation_times.sort()
            for i, value in enumerate(queue.eval_sorted_array(evaluation_times)):
                evaluated_queues[e_id][evaluation_times[i]] = value
            for i, value in enumerate(
                edge_loads[e_id].eval_sorted_array(evaluation_times)
            ):
                evaluated_loads[e_id][evaluation_times[i]] = value

        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([0.0], [0.0], 0.0, 0.0)
        assert len(edges) == len(flow.queues)

        result_predictions = []

        raw_predictions_input = np.array(
            [
                [
                    prediction_time,
                    *[
                        evaluated_queues[edge_id][
                            prediction_time + t * self._prediction_interval
                        ]
                        for edge_id in range(len(flow.queues))
                        if self._input_mask[edge_id]
                        for t in range(-self._past_timesteps + 1, 1)
                    ],
                    *[
                        evaluated_loads[edge_id][
                            prediction_time + t * self._prediction_interval
                        ]
                        for edge_id in range(len(flow.queues))
                        if self._input_mask[edge_id]
                        for t in range(-self._past_timesteps + 1, 1)
                    ],
                ]
                for prediction_time in prediction_times
            ]
        )
        future_queues_raw = self._model.predict(raw_predictions_input, verbose=0)
        future_queues_raw = np.maximum(
            future_queues_raw, np.zeros_like(future_queues_raw)
        )

        for prediction_ind, prediction_time in enumerate(prediction_times):
            queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)

            id_in_output_mask = -1
            times = [
                prediction_time + t * self._prediction_interval
                for t in range(0, self._future_timesteps + 1)
            ]
            for e_id, queue in enumerate(flow.queues):
                if not self._output_mask[e_id]:
                    queues[e_id] = zero_fct
                    continue
                id_in_output_mask += 1

                new_values = [
                    evaluated_queues[e_id][prediction_time],
                    *future_queues_raw[
                        prediction_ind,
                        id_in_output_mask
                        * self._future_timesteps : (id_in_output_mask + 1)
                        * self._future_timesteps,
                    ],
                ]

                for i in range(1, len(new_values)):
                    new_values[i] = max(
                        new_values[i],
                        new_values[i - 1]
                        - self._prediction_interval * self._network.capacity[e_id],
                    )

                queues[e_id] = PiecewiseLinear(times, new_values, 0.0, 0.0)

            result_predictions.append(queues)
        return result_predictions

    @staticmethod
    def from_model(
        network: Network,
        model_path: str,
        input_mask: np.ndarray,
        output_mask: np.ndarray,
        past_timesteps: int,
        future_timesteps: int,
        prediction_interval: float,
    ):
        model = tf.keras.models.load_model(model_path)
        return TFFullNetPredictor(
            model,
            input_mask,
            output_mask,
            network,
            past_timesteps,
            future_timesteps,
            prediction_interval,
        )
