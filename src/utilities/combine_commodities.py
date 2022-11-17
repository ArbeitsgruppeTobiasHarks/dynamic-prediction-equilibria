from typing import Dict, Set, Tuple
from core.graph import Node
from core.network import Commodity, Network

from core.predictors.predictor_type import PredictorType


def combine_commodities_with_same_sink(network: Network, exceptions: Set[Commodity] = set()):
    groups: Dict[Tuple[PredictorType, Node], Set[Commodity]] = {}
    for commodity in network.commodities:
        if commodity in exceptions:
            continue
        key = (commodity.predictor_type, commodity.sink)
        if key not in groups:
            groups[key] = {commodity}
        else:
            groups[key].add(commodity)
    network.commodities = []
    for key, group in groups.items():
        sources = {}
        for commodity in group:
            for source, value in commodity.sources.items():
                if source not in sources:
                    sources[source] = value
                else:
                    sources[source] += value
        network.commodities.append(Commodity(
            sources, key[1], key[0]
        ))
    for commodity in exceptions:
        network.commodities.append(commodity)
