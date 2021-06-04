from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from core.network import Network
from utilities.interpolate import LinearlyInterpolatedFunction


class Predictor(ABC):
    network: Network

    def __init__(self, network: Network):
        self.network = network

    @abstractmethod
    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        pass

    @abstractmethod
    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        pass

    @abstractmethod
    def type(self) -> str:
        pass


@dataclass
class PredictionResult:
    times: List[float]
    queues: List[np.ndarray]
