from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.interpolate import LinearlyInterpolatedFunction


class ConstantPredictor(Predictor):

    def is_constant(self) -> bool:
        return True

    def type(self) -> str:
        return "Constant Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        return PredictionResult(
            [times[-1], times[-1] + 1],
            [old_queues[-1], old_queues[-1]]
        )

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        queues = np.array([max(0., queue(phi)) for queue in old_queues])

        return PredictionResult(
            [phi, phi + 1],
            [queues, queues]
        )
