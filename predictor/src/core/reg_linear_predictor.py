from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.arrays import elem_rank


class RegularizedLinearPredictor(Predictor):

    def type(self) -> str:
        return "Regularized Linear Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        """
        This should return int_{phi-delta}^{phi} q_e(t) dt / delta
        """

        horizon = 2.
        delta = 1.
        phi = times[-1]
        m = len(self.network.graph.edges)

        rnk = elem_rank(times, phi - delta)
        integral = np.zeros(m)
        # First interval might be only partial...
        if phi - delta > 0:
            gradient = (old_queues[rnk + 1] - old_queues[rnk]) / (times[rnk + 1] - times[rnk])
            delta_time = times[rnk + 1] - (phi - delta)
            integral += delta_time * gradient
        rnk += 1
        while rnk < len(times) - 1:
            gradient = (old_queues[rnk + 1] - old_queues[rnk]) / (times[rnk + 1] - times[rnk])
            delta_time = times[rnk + 1] - times[rnk]
            integral += delta_time * gradient
            rnk += 1

        new_queues = old_queues[-1] + horizon * integral / delta

        return PredictionResult(
            [times[-1], times[-1] + horizon, times[-1] + horizon + 1],
            [old_queues[-1],
             new_queues,
             new_queues]
        )
