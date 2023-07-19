from __future__ import annotations

from typing import List, Optional

from core.dynamic_flow import DynamicFlow
from core.predictor import Predictor
from src.cython_test.piecewise_linear import PiecewiseLinear


class LinearRegressionPredictor(Predictor):
    def type(self) -> str:
        return "Kostas' Tokyo Weka Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(
        self, prediction_time: float, flow: DynamicFlow
    ) -> List[PiecewiseLinear]:
        t = prediction_time
        times = [t, t + 1, t + 2, t + 3, t + 4, t + 5, t + 6, t + 7, t + 8, t + 9]
        zero_fct = PiecewiseLinear([t], [0.0], 0.0, 0.0)
        queues: List[Optional[PiecewiseLinear]] = [None] * len(flow.queues)
        for i, old_queue in enumerate(flow.queues):
            queue_minus_0 = max(0.0, old_queue(t))
            queue_minus_1 = max(0.0, old_queue(t - 1))
            queue_minus_2 = max(0.0, old_queue(t - 2))
            queue_minus_3 = max(0.0, old_queue(t - 3))
            queue_minus_4 = max(0.0, old_queue(t - 4))

            if (
                max(
                    queue_minus_0,
                    queue_minus_1,
                    queue_minus_2,
                    queue_minus_3,
                    queue_minus_4,
                )
                == 0.0
            ):
                queues[i] = zero_fct
            else:
                queues[i] = PiecewiseLinear(
                    times,
                    [
                        queue_minus_0,
                        queue_minus_0 + 2.68,
                        1.01 * queue_minus_0 + 5.36,
                        1.01 * queue_minus_0 + 8.47,
                        1.02 * queue_minus_0 + 11.67,
                        1.02 * queue_minus_4 + 14.67,
                        1.02 * queue_minus_3 + 14.67,
                        1.02 * queue_minus_2 + 14.67,
                        1.02 * queue_minus_1 + 14.67,
                        1.02 * queue_minus_0 + 14.67,
                    ],
                    0.0,
                    0.0,
                )

        return queues
