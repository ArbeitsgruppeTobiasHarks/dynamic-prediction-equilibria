from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np

from core.graph import Node
from core.network import Network
from utilities.interpolate import LinearlyInterpolatedFunction


class Distributor(ABC):
    network: Network

    def __init__(self, network: Network):
        self.network = network

    @abstractmethod
    def distribute(self,
                   phi: float,
                   node_inflow: Dict[Node, float],
                   sink: Node,
                   queues: Optional[np.ndarray],
                   labels: Dict[Node, LinearlyInterpolatedFunction],
                   costs: List[LinearlyInterpolatedFunction]) -> Dict[int, float]:
        pass

    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def supports_const(self) -> bool:
        pass

    @abstractmethod
    def distribute_const(self,
                         phi: float,
                         node_inflow: Dict[Node, float],
                         sink: Node,
                         queues: Optional[np.ndarray],
                         labels: Dict[Node, float],
                         costs: np.ndarray) -> Dict[int, float]:
        pass
