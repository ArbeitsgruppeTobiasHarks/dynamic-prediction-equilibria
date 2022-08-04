from __future__ import annotations

from typing import List

from core.dynamic_flow import DynamicFlow

from core.predictor import ComputeMode, Predictor
from utilities.piecewise_linear import PiecewiseLinear


class ZeroPredictor(Predictor):

    def compute_mode(self) -> ComputeMode:
        return ComputeMode.CONSTANT_AND_SAME_FOR_SINK

    def type(self) -> str:
        return "Zero Predictor"

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        return [zero_fct for _ in flow.queues]
