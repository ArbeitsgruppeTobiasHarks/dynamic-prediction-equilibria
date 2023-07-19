from __future__ import annotations
import array

from typing import List

from core.dynamic_flow import DynamicFlow
from core.predictor import Predictor
from src.cython_test.piecewise_linear import PiecewiseLinear


class ZeroPredictor(Predictor):
    def is_constant(self) -> bool:
        return True

    def type(self) -> str:
        return "Zero Predictor"

    def predict(
        self, prediction_time: float, flow: DynamicFlow
    ) -> List[PiecewiseLinear]:
        zero_fct = PiecewiseLinear(array.array("d", [prediction_time]), array.array("d", [0.0]), 0.0, 0.0)
        return [zero_fct for _ in flow.queues]
