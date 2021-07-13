from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.piecewise_linear import PiecewiseLinear


class ZeroPredictor(Predictor):

    def is_constant(self) -> bool:
        return True

    def type(self) -> str:
        return "Zero Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        queues = np.zeros(len(old_queues[-1]))
        return PredictionResult(
            [times[-1], times[-1] + 1],
            [queues, queues]
        )

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) -> \
            List[PiecewiseLinear]:
        zero_fct = PiecewiseLinear([phi, phi + 1], [0., 0.])
        return [zero_fct for _ in old_queues]
