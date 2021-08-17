from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.arrays import elem_rank
from utilities.piecewise_linear import PiecewiseLinear


class RegularizedLinearPredictor(Predictor):
    horizon: float
    delta: float

    def __init__(self, network: Network, horizon: float, delta: float):
        super(RegularizedLinearPredictor, self).__init__(network)
        self.horizon = horizon
        self.delta = delta

    def is_constant(self) -> bool:
        return False

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) \
            -> List[PiecewiseLinear]:

        times = [phi, phi + self.horizon]
        phi_minus_delta = phi - self.delta
        queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
        for i, old_queue in enumerate(old_queues):
            queue_at_phi = max(0., old_queue(phi))
            queue_at_phi_minus_delta = max(0., old_queue(phi_minus_delta))
            new_queue = queue_at_phi + self.horizon * (queue_at_phi - queue_at_phi_minus_delta) / self.delta
            queues[i] = PiecewiseLinear(times, [queue_at_phi, new_queue], 0., 0.)

        return queues

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
