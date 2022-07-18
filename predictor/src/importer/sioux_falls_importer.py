import os
import pathlib
import random
from typing import Tuple, Callable
from math import pi

import pandas as pd

from core.network import Network, Commodity
from core.predictors.predictor_type import PredictorType
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


def _natural_earth_projection(latInRad: float, lngInRad: float) -> Tuple[float, float]:
    l = 0.870700 - 0.131979*latInRad**2 - 0.013791 * \
        latInRad**4 + 0.003971*latInRad**10-0.001529*latInRad**12
    d = latInRad * (1.007226 + 0.015085*latInRad**2 - 0.044475 *
                    latInRad**6 + 0.028874*latInRad**8 - 0.005916*latInRad**10)

    x = l * lngInRad
    y = d
    return (x * 1000 + 1333, - y * 1000 + 768)


def import_sioux_falls(edges_file_path: str, nodes_file_path: str, out_file_path: str, inflow_horizon: float) -> Network:
    net = pd.read_csv(edges_file_path, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    network = Network()
    #  columns: init_node, term_node, capacity, length, free_flow_time, b, power, speed, toll, link_type
    for _, e in net.iterrows():
        network.add_edge(e["init_node"], e["term_node"],
                         e["free_flow_time"], e["capacity"])

    nodes = pd.read_csv(nodes_file_path, sep='\t')
    trimmed = [s.strip().lower() for s in nodes.columns]
    nodes.columns = trimmed
    nodes.drop([';'], axis=1, inplace=True)
    network.graph.positions = {
        v["node"]: _natural_earth_projection(
            v["y"] / 180 * pi, v["x"] / 180 * pi)
        for _, v in nodes.iterrows()
    }
    os.makedirs(pathlib.Path(out_file_path).parent, exist_ok=True)
    network.to_file(out_file_path)
    return network


if __name__ == '__main__':
    import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/random-demands.pickle",
        float('inf'),
        demands_range_builder=lambda net: (
            min(net.capacity), max(net.capacity))
    )
