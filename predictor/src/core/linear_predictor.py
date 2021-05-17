from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult


class LinearPredictor(Predictor):

    def type(self) -> str:
        return "Linear Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        horizon = 2
        if len(old_queues) < 2:
            return PredictionResult(
                [times[-1], times[-1] + 1],
                [old_queues[-1], old_queues[-1]]
            )
        new_queues = np.maximum(
            np.zeros(len(self.network.graph.edges)),
            old_queues[-1] + horizon * (old_queues[-1] - old_queues[-2]) / (times[-1] - times[-2])
        )
        return PredictionResult(
            [times[-1], times[-1] + horizon, times[-1] + horizon + 1],
            [old_queues[-1],
             new_queues,
             new_queues]
        )
