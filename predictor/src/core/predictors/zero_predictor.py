from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.interpolate import LinearlyInterpolatedFunction


class ZeroPredictor(Predictor):

    def type(self) -> str:
        return "Zero Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        queues = np.zeros(len(old_queues[-1]))
        return PredictionResult(
            [times[-1], times[-1] + 1],
            [queues, queues]
        )

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        queues = np.zeros(len(old_queues))

        return PredictionResult(
            [phi, phi + 1],
            [queues, queues]
        )
