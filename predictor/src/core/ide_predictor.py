from __future__ import annotations
from abc import abstractmethod
from typing import List

import numpy as np

from core.predictor import Predictor, PredictionResult


class IDEPredictor(Predictor):

    @abstractmethod
    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        return PredictionResult(
            [times[-1]],
            [old_queues[-1]]
        )

