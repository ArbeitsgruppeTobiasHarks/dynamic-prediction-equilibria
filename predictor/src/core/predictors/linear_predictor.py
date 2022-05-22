from __future__ import annotations

from typing import List, Optional

import numpy as np
from core.dynamic_flow import DynamicFlow

from core.network import Network
from core.predictor import Predictor
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

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        times = [prediction_time, prediction_time + self.horizon]
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)
        for i, old_queue in enumerate(flow.queues):
            curr_queue = max(0., old_queue(prediction_time))
            gradient = old_queue.gradient(elem_rank(old_queue.times, prediction_time))
            new_queue = max(0., curr_queue + self.horizon * gradient)
            queues[i] = PiecewiseLinear(times, [curr_queue, new_queue], 0., 0.)

        return queues
