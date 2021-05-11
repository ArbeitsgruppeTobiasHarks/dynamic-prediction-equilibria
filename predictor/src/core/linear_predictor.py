from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult


class LinearPredictor(Predictor):

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        if len(old_queues) < 2:
            return PredictionResult(
                [times[-1], times[-1] + 1],
                [old_queues[-1], old_queues[-1]]
            )
        return PredictionResult(
            [times[-1], times[-1] + 2, times[-1] + 3],
            [old_queues[-1],
             old_queues[-1] + 2 * (old_queues[-1] - old_queues[-2]),
             old_queues[-1] + 2 * (old_queues[-1] - old_queues[-2])]
        )
