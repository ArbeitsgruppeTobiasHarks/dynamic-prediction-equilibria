from __future__ import annotations
import array

from typing import List, Optional

from core.dynamic_flow import DynamicFlow
from core.predictor import Predictor
from src.cython_test.piecewise_linear import PiecewiseLinear


class ConstantPredictor(Predictor):
    def is_constant(self) -> bool:
        return True

    def type(self) -> str:
        return "Constant Predictor"

    def predict(
        self, prediction_time: float, flow: DynamicFlow
    ) -> List[PiecewiseLinear]:
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)
        times = array.array("d", [prediction_time])
        for i, queue in enumerate(flow.queues):
            curr_queue = queue(prediction_time)
            queues[i] = PiecewiseLinear(times, array.array("d", [curr_queue]), 0.0, 0.0)

        return queues
