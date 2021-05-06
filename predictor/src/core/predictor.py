from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from core.network import Network


class Predictor(ABC):
    network: Network

    def __init__(self, network: Network):
        self.network = network

    @abstractmethod
    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        pass


@dataclass
class PredictionResult:
    times: List[float]
    queues: List[np.ndarray]
