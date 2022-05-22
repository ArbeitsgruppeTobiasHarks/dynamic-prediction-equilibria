from __future__ import annotations

from typing import List

import numpy as np
from core.dynamic_flow import DynamicFlow

from core.predictor import Predictor
from utilities.piecewise_linear import PiecewiseLinear


class ZeroPredictor(Predictor):

    def is_constant(self) -> bool:
        return True

    def type(self) -> str:
        return "Zero Predictor"

    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        zero_fct = PiecewiseLinear([prediction_time], [0.], 0., 0.)
        return [zero_fct for _ in flow.queues]
