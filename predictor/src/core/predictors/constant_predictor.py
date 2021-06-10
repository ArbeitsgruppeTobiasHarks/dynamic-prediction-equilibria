from __future__ import annotations

from typing import List, Optional

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

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> \
            List[LinearlyInterpolatedFunction]:
        queues: List[Optional[LinearlyInterpolatedFunction]] = [None] * len(old_queues)
        times = [phi, phi + 1]
        for i, queue in enumerate(old_queues):
            curr_queue = queue(phi)
            queues[i] = LinearlyInterpolatedFunction(times, [curr_queue, curr_queue])

        return queues
