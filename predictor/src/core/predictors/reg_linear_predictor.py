from __future__ import annotations

from typing import List, Optional

import numpy as np
from core.dynamic_flow import DynamicFlow

from core.network import Network
from core.predictor import ComputeMode, Predictor
from utilities.arrays import elem_rank
from utilities.piecewise_linear import PiecewiseLinear


class RegularizedLinearPredictor(Predictor):
    horizon: float
    delta: float

    def __init__(self, network: Network, horizon: float, delta: float):
        super(RegularizedLinearPredictor, self).__init__(network)
        self.horizon = horizon
        self.delta = delta

    def compute_mode(self) -> ComputeMode:
        return ComputeMode.DYNAMIC

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time, prediction_time + self.horizon]
        phi_minus_delta = prediction_time - self.delta
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)
        for i, old_queue in enumerate(flow.queues):
            queue_at_phi = max(0., old_queue(prediction_time))
            queue_at_phi_minus_delta = max(0., old_queue(phi_minus_delta))
            new_queue = queue_at_phi + self.horizon * (queue_at_phi - queue_at_phi_minus_delta) / self.delta
            queues[i] = PiecewiseLinear(times, [queue_at_phi, new_queue], 0., 0.)

        return queues

    def type(self) -> str:
        return "Regularized Linear Predictor"
