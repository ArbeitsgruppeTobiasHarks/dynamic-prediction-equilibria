from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.network import Network
from core.predictor import Predictor, PredictionResult
from utilities.arrays import elem_rank
from utilities.piecewise_linear import PiecewiseLinear


class LinearPredictor(Predictor):
    horizon: float

    def __init__(self, network: Network, horizon: float):
        super(LinearPredictor, self).__init__(network)
        self.horizon = horizon

    def type(self) -> str:
        return "Linear Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        if len(old_queues) < 2:
            return PredictionResult(
                [times[-1], times[-1] + 1],
                [old_queues[-1], old_queues[-1]]
            )
        new_queues = np.maximum(
            np.zeros(len(self.network.graph.edges)),
            old_queues[-1] + self.horizon * (old_queues[-1] - old_queues[-2]) / (times[-1] - times[-2])
        )
        return PredictionResult(
            [times[-1], times[-1] + self.horizon, times[-1] + self.horizon + 1],
            [old_queues[-1],
             new_queues,
             new_queues]
        )

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) \
            -> List[PiecewiseLinear]:
        times = [phi, phi + self.horizon, phi + self.horizon + 1]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
        for i, old_queue in enumerate(old_queues):
            curr_queue = max(0., old_queue(phi))
            gradient = old_queue.gradient(elem_rank(old_queue.times, phi))
            new_queue = max(0., curr_queue + self.horizon * gradient)
            queues[i] = PiecewiseLinear(times, [curr_queue, new_queue, new_queue])

        return queues
