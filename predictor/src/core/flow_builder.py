from __future__ import annotations

from core.dynamic_flow import PartialDynamicFlow
from core.network import Network
from core.predictor import Predictor, PredictionResult


class FlowBuilder:
    network: Network
    predictor: Predictor

    def __init__(self, network: Network, predictor: Predictor):
        self.network = network
        self.predictor = predictor

    def build_flow(self):
        flow = PartialDynamicFlow(self.network)
        phi = flow.times[-1]
        while phi < float('inf'):

            # TODO

            # Calculate dynamic shortest paths

            # Determine a good flow-split on active outgoing edges

            yield flow

    def _estimate_cost(self, flow: PartialDynamicFlow, prediction: PredictionResult):
        pass
