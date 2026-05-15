from typing import Dict, List, Set, Tuple

from dynflows.core.graph import Node
from dynflows.core.network import Commodity, Network
from dynflows.core.predictors.predictor_type import PredictorType


def combine_commodities_with_same_sink(
    network: Network,
    predictor_types: List[PredictorType] | None = None,
    exceptions: Set[Commodity] = set(),
):
    groups: Dict[Tuple[PredictorType, Node], Set[Commodity]] = {}
    for comm_idx, commodity in enumerate(network.commodities):
        if commodity in exceptions:
            continue
        key = (predictor_types and predictor_types[comm_idx], commodity.sink)
        if key not in groups:
            groups[key] = {commodity}
        else:
            groups[key].add(commodity)
    network.commodities = []
    if predictor_types is not None:
        predictor_types.clear()
    for key, group in groups.items():
        sources = {}
        for commodity in group:
            for source, value in commodity.sources.items():
                if source not in sources:
                    sources[source] = value
                else:
                    sources[source] += value
        network.commodities.append(Commodity(sources, key[1]))
        if predictor_types is not None:
            predictor_types.append(key[0])
    for commodity in exceptions:
        network.commodities.append(commodity)
