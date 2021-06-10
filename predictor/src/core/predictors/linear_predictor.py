from __future__ import annotations

from typing import List

import numpy as np

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.arrays import elem_rank
from utilities.interpolate import LinearlyInterpolatedFunction


class LinearPredictor(Predictor):
    horizon: float

    def __init__(self, network: Network, horizon: float):
        super(LinearPredictor, self).__init__(network)
        self.horizon = horizon

    def type(self) -> str:
        return "Linear Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        if len(old_queues) < 2:
            return PredictionResult(
                [times[-1], times[-1] + 1],
                [old_queues[-1], old_queues[-1]]
            )
        new_queues = np.maximum(
            np.zeros(len(self.network.graph.edges)),
            old_queues[-1] + self.horizon * (old_queues[-1] - old_queues[-2]) / (times[-1] - times[-2])
        )
        return PredictionResult(
            [times[-1], times[-1] + self.horizon, times[-1] + self.horizon + 1],
            [old_queues[-1],
             new_queues,
             new_queues]
        )

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        queues = np.array([max(0., queue(phi)) for queue in old_queues])
        gradients = np.array([queue.gradient(elem_rank(queue.times, phi)) for queue in old_queues])
        new_queues = np.maximum(queues + self.horizon * gradients, np.zeros(len(old_queues)))

        return PredictionResult(
            [phi, phi + self.horizon, phi + self.horizon + 1],
            [queues, new_queues, new_queues]
        )
