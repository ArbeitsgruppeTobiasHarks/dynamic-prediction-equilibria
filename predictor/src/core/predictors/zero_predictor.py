from __future__ import annotations

from typing import List

import numpy as np
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow

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

    def predict(self, prediction_time: float, flow: MultiComPartialDynamicFlow) -> List[PiecewiseLinear]:
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        return [zero_fct for _ in flow.queues]
