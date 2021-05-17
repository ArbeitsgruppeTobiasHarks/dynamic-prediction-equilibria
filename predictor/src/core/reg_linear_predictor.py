from __future__ import annotations

from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.arrays import elem_rank


class RegularizedLinearPredictor(Predictor):

    def type(self) -> str:
        return "Regularized Linear Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        horizon = 2.
        delta = 1.
        phi = times[-1]
        m = len(self.network.graph.edges)

        rnk = elem_rank(times, phi - delta)
        if phi - delta > 0:
            queue_at_phi_minus_delta = old_queues[rnk] \
                                       + (phi - delta - times[rnk]) * \
                                       (old_queues[rnk + 1] - old_queues[rnk]) / (times[rnk + 1] - times[rnk])
        else:
            queue_at_phi_minus_delta = np.zeros(m)
        new_queues = np.maximum(
            np.zeros(len(self.network.graph.edges)),
            old_queues[-1] + horizon * (old_queues[-1] - queue_at_phi_minus_delta) / delta
        )

        return PredictionResult(
            [times[-1], times[-1] + horizon, times[-1] + horizon + 1],
            [old_queues[-1],
             new_queues,
             new_queues]
        )
