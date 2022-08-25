from typing import Tuple, Callable
from math import pi

import pandas as pd

from core.network import Network
from core.predictors.predictor_type import PredictorType
from scenarios.scenario_utils import get_demand_with_inflow_horizon


DemandsRangeBuilder = Callable[[Network], Tuple[float, float]]


def natural_earth_projection(latInRad: float, lngInRad: float) -> Tuple[float, float]:
    l = 0.870700 - 0.131979*latInRad**2 - 0.013791 * \
        latInRad**4 + 0.003971*latInRad**10-0.001529*latInRad**12
    d = latInRad * (1.007226 + 0.015085*latInRad**2 - 0.044475 *
                    latInRad**6 + 0.028874*latInRad**8 - 0.005916*latInRad**10)

    x = l * lngInRad
    y = d
    return (x * 1000 + 1333, - y * 1000 + 768)


def add_commodities(network: Network, trips_tntp_file_path: str, inflow_horizon: float):
    with open(trips_tntp_file_path, 'r') as file:
        all_rows = file.read()

    blocks = all_rows.split('Origin')[1:]
    od_pairs = []
    for block in blocks:
        lines = block.replace(";", "\n").split('\n')
        origin = int(lines[0])
        destination_lines = lines[1:]

        for line in destination_lines:
            if len(line.strip()) == 0:
                continue
            entries = line.split(":")
            assert len(entries) == 2
            destination = int(entries[0])
            demand = float(entries[1])
            od_pairs.append((origin, destination, demand))

    for (origin, destination, demand) in od_pairs:
        network.add_commodity(
            origin, destination, get_demand_with_inflow_horizon(demand, inflow_horizon), PredictorType.CONSTANT)


def add_node_positions(network: Network, node_tntp_file_path: str):
    nodes = pd.read_csv(node_tntp_file_path, sep='\t', skip_blank_lines=True)
    nodes.columns = [s.strip().lower() for s in nodes.columns]
    nodes.drop([';'], axis=1, inplace=True)
    network.graph.positions = {
        int(v["node"]): natural_earth_projection(
            float(v["y"]) / 180 * pi, float(v["x"]) / 180 * pi)
        for _, v in nodes.iterrows()
    }


def import_network(edges_file_path: str) -> Network:
    net = pd.read_csv(edges_file_path, skiprows=8,
                      sep='\t', skip_blank_lines=True)
    net.columns = [s.strip().lower() for s in net.columns]
    net.drop(['~', ';'], axis=1, inplace=True)
    network = Network()
    #  columns: init_node, term_node, capacity, length, free_flow_time, b, power, speed, toll, link_type
    for _, e in net.iterrows():
        network.add_edge(int(e["init_node"]), int(e["term_node"]),
                         e["free_flow_time"], e["capacity"])
    return network
