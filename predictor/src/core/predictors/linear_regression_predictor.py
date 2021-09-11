from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.predictor import Predictor, PredictionResult
from utilities.piecewise_linear import PiecewiseLinear


class LinearRegressionPredictor(Predictor):

    def type(self) -> str:
        return "Linear Regression Predictor"

    def is_constant(self) -> bool:
        return False

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        raise NotImplementedError()

    def predict_from_fcts(self, old_queues: List[PiecewiseLinear], phi: float) -> List[PiecewiseLinear]:
        times = [phi, phi + 1, phi + 2, phi + 3, phi + 4, phi + 5, phi + 6, phi + 7, phi + 8, phi + 9]
        zero_fct = PiecewiseLinear([phi], [0.], 0., 0.)
        queues: List[Optional[PiecewiseLinear]] = [None] * len(old_queues)
        for i, old_queue in enumerate(old_queues):
            queue_minus_0 = max(0., old_queue(phi))
            queue_minus_1 = max(0., old_queue(phi - 1))
            queue_minus_2 = max(0., old_queue(phi - 2))
            queue_minus_3 = max(0., old_queue(phi - 3))
            queue_minus_4 = max(0., old_queue(phi - 4))

            if max(queue_minus_0, queue_minus_1, queue_minus_2, queue_minus_3, queue_minus_4) == 0.:
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
                        1.02 * queue_minus_0 + 14.67
                    ],
                    0., 0.
                )

        return queues
