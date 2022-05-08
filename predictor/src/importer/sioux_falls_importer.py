import os
import pathlib
import random
from typing import Tuple, Callable

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


def _add_commodity(network: Network, inflow_horizon: float, demands_range: Tuple[float, float]):
    source = network.graph.nodes[1]
    sink = network.graph.nodes[24]
    demand = random.uniform(*demands_range)
    if inflow_horizon < float('inf'):
        commodity = Commodity(source, sink, RightConstant([0., inflow_horizon], [demand, 0.], (0, float('inf'))),
                                PredictorType.CONSTANT)
    else:
        commodity = Commodity(source, sink, RightConstant([0.], [demand], (0, float('inf'))),
                                PredictorType.CONSTANT)
    network.commodities.append(commodity)


DemandsRangeBuilder = Callable[[Network], Tuple[float, float]]


def import_sioux_falls(edges_file_path: str, nodes_file_path: str, out_file_path: str, inflow_horizon: float,
                       demands_range_builder: DemandsRangeBuilder) -> Network:
    net = pd.read_csv(edges_file_path, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    network = Network()
    #  columns: init_node, term_node, capacity, length, free_flow_time, b, power, speed, toll, link_type
    for _, e in net.iterrows():
        network.add_edge(e["init_node"], e["term_node"], e["free_flow_time"], e["capacity"])
    
    
    nodes = pd.read_csv(nodes_file_path, sep='\t')
    trimmed = [s.strip().lower() for s in nodes.columns]
    nodes.columns = trimmed
    nodes.drop([';'], axis=1, inplace=True)
    network.graph.positions = {
        v["node"]: (v["x"], v["y"])
        for _, v in nodes.iterrows()
    }

    
    random.seed(-3)
    _add_commodity(network, inflow_horizon, demands_range_builder(network))
    network.remove_unnecessary_nodes()
    network.print_info()
    os.makedirs(pathlib.Path(out_file_path).parent, exist_ok=True)
    network.to_file(out_file_path)
    return network


if __name__ == '__main__':
    import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/random-demands.pickle",
        float('inf'),
        demands_range_builder=lambda net: (min(net.capacity), max(net.capacity))
    )
