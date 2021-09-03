import os
import pathlib
import random

import pandas as pd

from core.network import Network, Commodity
from core.predictors.predictor_type import PredictorType
from utilities.right_constant import RightConstant


def _generate_commodities(network: Network, number: int, inflow_horizon: float):
    assert number < len(network.graph.nodes)
    commodities = []
    nodes = list(network.graph.nodes.values())
    while len(network.commodities) < number:
        source = random.choice(nodes)
        sink = random.choice(nodes)
        if sink not in network.graph.get_reachable_nodes(source):
            continue
        demand = random.randint(4500, 10000)
        if inflow_horizon < float('inf'):
            commodity = Commodity(source, sink, RightConstant([0., inflow_horizon], [demand, 0.], (0, float('inf'))), PredictorType.CONSTANT)
        else:
            commodity = Commodity(source, sink, RightConstant([0.], [demand], (0, float('inf'))), PredictorType.CONSTANT)
        network.commodities.append(commodity)
    return commodities


def import_sioux_falls(file_path: str, out_file_path: str, inflow_horizon: float):
    net = pd.read_csv(file_path, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    network = Network()
    #  columns: init_node, term_node, capacity, length, free_flow_time, b, power, speed, toll, link_type
    for _, e in net.iterrows():
        network.add_edge(e["init_node"], e["term_node"], e["free_flow_time"], e["capacity"])
    _generate_commodities(network, len(network.graph.nodes) // 2, inflow_horizon)
    network.remove_unnecessary_nodes()
    network.print_info()
    os.makedirs(pathlib.Path(out_file_path).parent, exist_ok=True)
    network.to_file(out_file_path)


if __name__ == '__main__':
    import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/random-demands.pickle",
        float('inf')
    )
