from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from core.dynamic_flow import DynamicFlow

from core.network import Network
from utilities.piecewise_linear import PiecewiseLinear


class Predictor(ABC):
    network: Network

    def __init__(self, network: Network):
        self.network = network

    @abstractmethod
    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        pass

    @abstractmethod
    def is_constant(self) -> bool:
        pass

    @abstractmethod
    def type(self) -> str:
        pass
