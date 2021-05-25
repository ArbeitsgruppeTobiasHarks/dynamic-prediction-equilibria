from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.interpolate import LinearlyInterpolatedFunction


class LinearRegressionPredictor(Predictor):

    def type(self) -> str:
        return "Constant Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        raise NotImplementedError()

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        queues_minus_0 = np.array([queue(phi) for queue in old_queues])
        queues_minus_1 = np.array([queue(phi - 1) for queue in old_queues])
        queues_minus_2 = np.array([queue(phi - 2) for queue in old_queues])
        queues_minus_3 = np.array([queue(phi - 3) for queue in old_queues])
        queues_minus_4 = np.array([queue(phi - 4) for queue in old_queues])

        return PredictionResult(
            [phi, phi + 1, phi + 2, phi + 3, phi + 4, phi + 5, phi + 6, phi + 7, phi + 8, phi + 9, phi + 10],
            [
                queues_minus_0,
                queues_minus_0 + 2.68,
                1.01 * queues_minus_0 + 5.36,
                1.01 * queues_minus_0 + 8.47,
                1.02 * queues_minus_0 + 11.67,
                1.02 * queues_minus_4 + 14.67,
                1.02 * queues_minus_3 + 14.67,
                1.02 * queues_minus_2 + 14.67,
                1.02 * queues_minus_1 + 14.67,
                1.02 * queues_minus_0 + 14.67,
                1.02 * queues_minus_0 + 14.67
             ]
        )
