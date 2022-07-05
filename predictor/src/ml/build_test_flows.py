from math import ceil, log10
import os
import pickle
import random
from typing import Optional

from core.flow_builder import FlowBuilder
from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.predictor_type import PredictorType
from eval.evaluate import COLORS
from utilities.build_with_times import build_with_times
from utilities.right_constant import RightConstant
from utilities.file_lock import wait_for_locks, with_file_lock
from visualization.to_json import to_visualization_json


def generate_network_demands(network: Network, random_seed: int, inflow_horizon: float, sigma: Optional[float] = None):
    if sigma is None:
        sigma = min(network.capacity) / 2.
    random.seed(random_seed)
    for commodity in network.commodities:
        demand = max(0., random.gauss(commodity.net_inflow.values[0], sigma))
        if inflow_horizon < float('inf'):
            commodity.net_inflow = RightConstant(
                [0., inflow_horizon], [demand, 0.], (0., float('inf')))
        else:
            commodity.net_inflow = RightConstant(
                [0.], [demand], (0., float('inf')))


def build_flows(network_path: str, out_dir: str, inflow_horizon: float, number_flows: int, horizon: float, reroute_interval: float,
                demand_sigma: Optional[float] = None, check_for_optimizations: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    if number_flows == 0:
        return
    print()
    print("You can start multiple processes with this command to speed up the generation.")
    print("We will only generate flows that are not yet saved to disk yet.")
    print()
    for flow_id in range(number_flows):
        flow_path = os.path.join(
            out_dir, f"{str(flow_id).zfill(ceil(log10(number_flows)))}.flow.pickle")

        def handle(open_file):
            network = Network.from_file(network_path)
            generate_network_demands(
                network, flow_id, inflow_horizon, sigma=demand_sigma)
            print(f"Generating flow with seed {flow_id}...")
            if check_for_optimizations:
                assert (lambda: False)(
                ), "Use PYTHONOPTIMIZE=TRUE for a faster generation."

            predictors = {PredictorType.CONSTANT: ConstantPredictor(network)}

            flow_builder = FlowBuilder(network, predictors, reroute_interval)
            flow, _ = build_with_times(
                flow_builder, flow_id, reroute_interval, horizon)

            print(f"Successfully built flow up to time {flow.phi}!")
            with open_file("wb") as file:
                pickle.dump(flow, file)
            to_visualization_json(flow_path + ".json", flow, network, {
                id: COLORS[comm.predictor_type] for (id, comm) in enumerate(network.commodities)})
            print(f"Successfully written flow to disk!\n\n")
        with_file_lock(flow_path, handle)

    wait_for_locks(out_dir)
