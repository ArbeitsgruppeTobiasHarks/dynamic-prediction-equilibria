from __future__ import annotations

from typing import List, Optional

from dynflows.core.dynamic_flow import DynamicFlow
from dynflows.core.machine_precision import eps
from dynflows.core.network import Network
from dynflows.core.predictor import Predictor
from dynflows.utilities.piecewise_linear import PiecewiseLinear


class RegularizedLinearPredictor(Predictor):
    horizon: float
    delta: float

    def __init__(self, network: Network, horizon: float, delta: float):
        super(RegularizedLinearPredictor, self).__init__(network)
        self.horizon = horizon
        self.delta = delta

    def is_constant(self) -> bool:
        return False

    def predict(
        self, prediction_time: float, flow: DynamicFlow
    ) -> List[PiecewiseLinear]:
        times = [prediction_time, prediction_time + self.horizon]
        phi_minus_delta = prediction_time - self.delta
        queues: List[PiecewiseLinear] = []
        for i, old_queue in enumerate(flow.queues):
            queue_at_phi = max(0.0, old_queue(prediction_time))
            queue_at_phi_minus_delta = max(0.0, old_queue(phi_minus_delta))
            gradient = (queue_at_phi - queue_at_phi_minus_delta) / self.delta
            new_queue = queue_at_phi + self.horizon * gradient

            if new_queue < 0 and queue_at_phi > eps:
                new_time = prediction_time - queue_at_phi / gradient
                queues.append(
                    PiecewiseLinear(
                        [prediction_time, new_time], [queue_at_phi, 0.0], 0.0, 0.0
                    )
                )
            else:
                queues.append(
                    PiecewiseLinear(times, [queue_at_phi, new_queue], 0.0, 0.0)
                )

        return queues

    def type(self) -> str:
        return "Regularized Linear Predictor"
