from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

from core.dynamic_flow import PartialDynamicFlow
from core.graph import Node
from core.network import Network
from utilities.interpolate import LinearlyInterpolatedFunction


class Distributor(ABC):
    network: Network

    def __init__(self, network: Network):
        self.network = network

    @abstractmethod
    def distribute(self,
                   flow: PartialDynamicFlow,
                   labels: Dict[Node, LinearlyInterpolatedFunction],
                   costs: List[LinearlyInterpolatedFunction]) -> np.ndarry:
        pass
