from __future__ import annotations

from typing import List

import numpy as np

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.arrays import elem_rank
from utilities.interpolate import LinearlyInterpolatedFunction


class RegularizedLinearPredictor(Predictor):
    horizon: float
    delta: float

    def __init__(self, network: Network, horizon: float, delta: float):
        super(RegularizedLinearPredictor, self).__init__(network)
        self.horizon = horizon
        self.delta = delta

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        phi_minus_delta = phi - self.delta
        queue_at_phi_minus_delta = np.asarray([queue(phi_minus_delta) for queue in old_queues])
        queue_at_phi = np.asarray([queue(phi) for queue in old_queues])

        new_queues = np.maximum(
            np.zeros(len(self.network.graph.edges)),
            queue_at_phi + self.horizon * (queue_at_phi - queue_at_phi_minus_delta) / self.delta
        )

        return PredictionResult(
            [phi, phi + self.horizon, phi + self.horizon + 1],
            [queue_at_phi, new_queues, new_queues]
        )

    def type(self) -> str:
        return "Regularized Linear Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        phi = times[-1]
        m = len(self.network.graph.edges)

        rnk = elem_rank(times, phi - self.delta)
        if phi - self.delta > 0:
            queue_at_phi_minus_delta = old_queues[rnk] \
                                       + (phi - self.delta - times[rnk]) * \
                                       (old_queues[rnk + 1] - old_queues[rnk]) / (times[rnk + 1] - times[rnk])
        else:
            queue_at_phi_minus_delta = np.zeros(m)

        new_queues = np.maximum(
            np.zeros(len(self.network.graph.edges)),
            old_queues[-1] + self.horizon * (old_queues[-1] - queue_at_phi_minus_delta) / self.delta
        )

        return PredictionResult(
            [times[-1], times[-1] + self.horizon, times[-1] + self.horizon + 1],
            [old_queues[-1],
             new_queues,
             new_queues]
        )
