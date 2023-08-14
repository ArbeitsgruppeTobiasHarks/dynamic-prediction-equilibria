import gzip
import os
import pickle
import random
from math import ceil, log10
from typing import Callable, Optional

from core.dynamic_flow import DynamicFlow
from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.predictor_type import PredictorType
from eval.evaluate import COLORS
from utilities.build_with_times import build_with_times
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.file_lock import no_op, wait_for_locks, with_file_lock
from utilities.right_constant import RightConstant
from visualization.to_json import merge_commodities, to_visualization_json

from ml.build_test_flows import generate_network_demands
from convergence.path_flow_builder import PathFlowBuilder

def build_path_flows(
    paths,
    network_path: str,
    out_dir: str,
    inflow_horizon: float,
    number_flows: int,
    horizon: float,
    reroute_interval: float,
    demand_sigma: Optional[float] = None,
    check_for_optimizations: bool = True,
    on_flow_computed: Callable[[str, DynamicFlow], None] = no_op,
    generate_visualization: bool = True,
    save_dummy: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    if number_flows == 0:
        return
    print()
    print(
        "You can start multiple processes with this command to speed up the generation."
    )
    print("We will only generate flows that are not yet saved to disk yet.")
    print()
    for flow_index in range(number_flows):
        flow_id = str(flow_index).zfill(ceil(log10(number_flows)))
        flow_path = os.path.join(out_dir, f"{flow_id}.flow.pickle")
        visualization_path = flow_path + ".json"

        def handle(open_file):
            network = Network.from_file(network_path)
            generate_network_demands(
                network, flow_index, inflow_horizon, sigma=demand_sigma
            )
            combine_commodities_with_same_sink(network)
            print(f"Generating flow with seed {flow_index}...")
            if check_for_optimizations:
                assert (
                    lambda: False
                )(), "Use PYTHONOPTIMIZE=TRUE for a faster generation."

            flow_builder = PathFlowBuilder(network, paths, reroute_interval)
            flow, _ = build_with_times(
                flow_builder, flow_index, reroute_interval, horizon
            )

            if save_dummy:
                with open(flow_path, "w") as file:
                    file.write("Dummy file")
                print(f"Written dummy file to disk!")
            else:
                with gzip.open(flow_path, "wb") as file:
                    pickle.dump(flow, file)

                print(f"Successfully written flow to disk!")

            on_flow_computed(flow_id, flow)

            if generate_visualization:
                merged_flow = merge_commodities(
                    flow, network, range(len(network.commodities))
                )

                to_visualization_json(
                    visualization_path,
                    merged_flow,
                    network,
                    {
                        id: COLORS[comm.predictor_type]
                        for (id, comm) in enumerate(network.commodities)
                    },
                )

                print(f"Successfully written visualization to disk!")

            print()

        expect_exists = [flow_path]
        if generate_visualization:
            expect_exists.append(visualization_path)
        with_file_lock(flow_path, handle, expect_exists)

    wait_for_locks(out_dir)
