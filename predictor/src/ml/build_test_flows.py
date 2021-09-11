import os
import pickle
import random

from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.predictor_type import PredictorType
from utilities.build_with_times import build_with_times
from utilities.right_constant import RightConstant


def generate_network_demands(network: Network, random_seed: int, inflow_horizon: float, sigma: float):
    random.seed(random_seed)
    for commodity in network.commodities:
        demand = max(0., random.gauss(commodity.net_inflow.values[0], sigma))
        if inflow_horizon < float('inf'):
            commodity.net_inflow = RightConstant([0., inflow_horizon], [demand, 0.], (0., float('inf')))
        else:
            commodity.net_inflow = RightConstant([0.], [demand], (0., float('inf')))


def build_flows(network_path: str, out_directory: str, number_flows: int, horizon: float, reroute_interval: float,
                check_for_optimizations: bool = True):
    os.makedirs(out_directory, exist_ok=True)
    print()
    print("You can start multiple processes with this command to speed up the generation.\n"
          "We will only generate flows that are not yet saved to disk yet.")
    print()
    for flow_id in range(number_flows):
        lock_path = os.path.join(out_directory, f".lock.{flow_id}.flow.pickle")
        flow_path = os.path.join(out_directory, f"{flow_id}.flow.pickle")
        if os.path.exists(lock_path):
            print(f"Detected lock file for flow#{flow_id}. Skipping...")
            continue
        elif os.path.exists(flow_path):
            print(f"Flow#{flow_id} was already built. Skipping...")
            continue

        with open(lock_path, "w") as file:
            file.write("")

        network = Network.from_file(network_path)
        generate_network_demands(network, flow_id, float('inf'), sigma=min(network.capacity) / 1000.)
        print(f"Generating flow with seed {flow_id}...")
        if check_for_optimizations:
            assert (lambda: False)(), "Use PYTHONOPTIMIZE=TRUE for a faster generation."

        predictors = {PredictorType.CONSTANT: ConstantPredictor(network)}

        flow_builder = MultiComFlowBuilder(network, predictors, reroute_interval)
        flow = build_with_times(flow_builder, flow_id, reroute_interval, horizon)

        print(f"Successfully built flow up to time {flow.phi}!")
        with open(flow_path, "wb") as file:
            pickle.dump(flow, file)
        os.remove(lock_path)
        print(f"Successfully written flow to disk!")
        print("\n")
