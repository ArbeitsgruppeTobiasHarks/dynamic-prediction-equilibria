from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult


class IDEPredictor(Predictor):

    def type(self) -> str:
        return "Constant Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        return PredictionResult(
            [times[-1], times[-1] + 1],
            [old_queues[-1], old_queues[-1]]
        )
