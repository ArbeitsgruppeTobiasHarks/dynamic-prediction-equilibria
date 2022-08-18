import random
from typing import Tuple, Callable
from math import pi

import pandas as pd

from core.network import Network, Commodity
from core.predictors.predictor_type import PredictorType
from importer.tntp_importer import import_network, natural_earth_projection
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.right_constant import RightConstant


def _generate_commodities(network: Network, number: int, inflow_horizon: float, demands_range: Tuple[float, float]):
    assert number < len(network.graph.nodes)
    commodities = []
    nodes = list(network.graph.nodes.values())
    while len(network.commodities) < number:
        source = random.choice(nodes)
        sink = random.choice(nodes)
        if sink not in network.graph.get_reachable_nodes(source):
            continue
        demand = random.uniform(*demands_range)
        if inflow_horizon < float('inf'):
            commodity = Commodity(source, sink, RightConstant([0., inflow_horizon], [demand, 0.], (0, float('inf'))),
                                  PredictorType.CONSTANT)
        else:
            commodity = Commodity(source, sink, RightConstant([0.], [demand], (0, float('inf'))),
                                  PredictorType.CONSTANT)
        network.commodities.append(commodity)
    return commodities


DemandsRangeBuilder = Callable[[Network], Tuple[float, float]]


def add_od_pairs(network: Network, od_pairs_file_path: str, inflow_horizon: float):
    od_pairs = pd.read_csv(od_pairs_file_path, header=0)
    for _, e in od_pairs.iterrows():
        network.add_commodity(int(e["O"]), int(e["D"]), get_demand_with_inflow_horizon(
            e["Ton"], inflow_horizon), PredictorType.CONSTANT)


def import_sioux_falls(edges_file_path: str, nodes_file_path: str) -> Network:
    network = import_network(edges_file_path)
    nodes = pd.read_csv(nodes_file_path, sep='\t')
    trimmed = [s.strip().lower() for s in nodes.columns]
    nodes.columns = trimmed
    nodes.drop([';'], axis=1, inplace=True)
    network.graph.positions = {
        v["node"]: natural_earth_projection(
            v["y"] / 180 * pi, v["x"] / 180 * pi)
        for _, v in nodes.iterrows()
    }
    return network


if __name__ == '__main__':
    import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/random-demands.pickle",
        float('inf'),
        demands_range_builder=lambda net: (
            min(net.capacity), max(net.capacity))
    )
