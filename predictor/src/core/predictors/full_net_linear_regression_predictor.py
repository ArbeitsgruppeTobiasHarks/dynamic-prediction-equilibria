from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from core.dynamic_flow import DynamicFlow

from core.network import Network
from core.predictor import Predictor, PredictionResult
from ml.DataLoader import QueueDataset
from utilities.piecewise_linear import PiecewiseLinear


class FullNetLinearRegressionPredictor(Predictor):

    def __init__(self, lin_reg: LinearRegression, test_mask: np.ndarray, network: Network):
        super().__init__(network)
        self._lin_reg = lin_reg
        self._test_mask = test_mask

    def type(self) -> str:
        return "Full Net Linear Regression Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time + t for t in range(0, 21, 1)]
        edges = self.network.graph.edges
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        assert len(edges) == len(flow.queues)
        input_times = [prediction_time - t for t in range(-20, 1)]
        inputs = np.array([
            [queue(t) for t in input_times] for queue in flow.queues
        ])[self._test_mask].flatten()
        prediction = self._lin_reg.predict([inputs])[0]
        prediction = np.maximum(prediction, np.zeros_like(prediction))
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)
        for e_id, old_queue in enumerate(flow.queues):
            if not self._test_mask[e_id]:
                queues[e_id] = zero_fct
                continue
            masked_id = np.count_nonzero(self._test_mask[:e_id])
            new_values = [inputs[21*masked_id + 20], *prediction[masked_id * 20: (masked_id + 1) * 20]]
            queues[e_id] = PiecewiseLinear(
                times,
                new_values,
                0., 0.
            )

        return queues

    @staticmethod
    def from_model(network: Network, model_path: str, queues_folder: str):
        with open(model_path, "rb") as file:
            lin_reg: LinearRegression = pickle.load(file)
        dataset = QueueDataset(queues_folder, 20, 20, network, False, 'cpu')
        return FullNetLinearRegressionPredictor(lin_reg, dataset.test_mask, network)
