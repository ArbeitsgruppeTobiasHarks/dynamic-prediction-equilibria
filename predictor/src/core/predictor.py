from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from core.dynamic_flow import DynamicFlow

from core.network import Network
from utilities.piecewise_linear import PiecewiseLinear


class ComputeMode(Enum):
    DYNAMIC = 0
    CONSTANT = 1
    CONSTANT_AND_SAME_FOR_SINK = 2


class Predictor(ABC):
    network: Network

    def __init__(self, network: Network):
        self.network = network

    @abstractmethod
    def predict(self, prediction_time: float, flow: DynamicFlow) -> List[PiecewiseLinear]:
        pass

    def batch_predict(self, prediction_times: List[float], flow: DynamicFlow) -> List[List[PiecewiseLinear]]:
        if flow.phi < max(prediction_times):
            raise ValueError("Prediction time larger than flow horizon.")
        return [self.predict(prediction_time, flow) for prediction_time in prediction_times]

    @abstractmethod
    def compute_mode(self) -> ComputeMode:
        pass

    @abstractmethod
    def type(self) -> str:
        pass
