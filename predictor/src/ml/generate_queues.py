import os
import pickle

import numpy as np
import pandas as pd
from math import floor

from core.dynamic_flow import DynamicFlow
from core.network import Network
from utilities.file_lock import wait_for_locks, with_file_lock


def generate_queues(past_timesteps: int, flows_folder: str, out_folder: str, horizon: int, step_length: int):
    os.makedirs(out_folder, exist_ok=True)
    files = [file for file in os.listdir(flows_folder) if
             file.endswith(".flow.pickle") and not file.startswith(".lock.")]

    for flow_path in files:
        flow_id = flow_path[: len(flow_path) - len(".flow.pickle")]
        queue_path = os.path.join(
            out_folder, f"{flow_id}-queues-complete.csv.gz")
        if os.path.exists(queue_path):
            print(f"Queue for flow#{flow_id} already exists. Skipping...")
            continue
        with open(os.path.join(flows_folder, flow_path), "rb") as file:
            flow = pickle.load(file)
        times = range(-past_timesteps, horizon + 1, step_length)
        queues = np.asarray([
            [queue(time) for time in times] for queue in flow.queues
        ])
        np.savetxt(queue_path, queues)


def generate_queues_and_edge_loads(past_timesteps: int, flows_dir: str, out_dir: str, horizon: int, reroute_interval: float, prediction_interval: float):
    os.makedirs(out_dir, exist_ok=True)

    reroute_intervals_in_prediction_interval = round(
        prediction_interval / reroute_interval)
    if abs(reroute_intervals_in_prediction_interval - prediction_interval / reroute_interval) > 1e-14:
        raise ValueError(
            "Prediction interval is not a multiple of reroute interval.")

    files = sorted([file for file in os.listdir(
        flows_dir) if file.endswith(".flow.pickle")])

    for flow_filename in files:
        flow_id = flow_filename[: len(flow_filename) - len(".flow.pickle")]
        out_path = os.path.join(
            out_dir, f"{flow_id}-queues-and-edge-loads.npy")

        def handle(_):
            print(f"Building queues for Flow#{flow_id}...")

            with open(os.path.join(flows_dir, flow_filename), "rb") as file:
                flow: DynamicFlow = pickle.load(file)
            times = [
                i*reroute_interval for i in range(-past_timesteps*reroute_intervals_in_prediction_interval, floor(horizon / reroute_interval) + 1)]
            edgeLoads = flow.get_edge_loads()
            edgeLoadSamples = np.asarray([
                [load(time) if time >= 0 else 0 for time in times] for load in edgeLoads
            ])
            queueSamples = np.asarray([
                [queue(time) for time in times] for queue in flow.queues
            ])
            stacked = np.stack([queueSamples, edgeLoadSamples])
            np.save(out_path, stacked)

        with_file_lock(out_path, handle)

    wait_for_locks(out_dir)


def generate_training_data_with_edge_loads(past_timesteps: int, future_timesteps: int, flows_dir: str, training_data_dir: str, horizon: float, reroute_interval: float, prediction_interval: float):
    os.makedirs(training_data_dir, exist_ok=True)
    files = [file for file in os.listdir(flows_dir) if
             file.endswith(".flow.pickle") and not file.startswith(".lock.")]

    for flow_path in files:
        flow_id = flow_path[: len(flow_path) - len(".flow.pickle")]
        out_path = os.path.join(
            training_data_dir, f"{flow_id}-training-data.npy")
        if os.path.exists(out_path):
            print(
                f"Training data for flow#{flow_id} already exists. Skipping...")
            continue
        with open(os.path.join(flows_dir, flow_path), "rb") as file:
            flow: DynamicFlow = pickle.load(file)

        prediction_times = [
            i*reroute_interval for i in range(floor(horizon / reroute_interval) + 1)
        ]

        edgeLoads = flow.get_edge_loads()

        training_sets = np.ndarray(
            (
                len(prediction_times),
                len(flow.queues) * past_timesteps +
                len(flow.queues) * past_timesteps +
                len(flow.queues) * future_timesteps
            )
        )

        row = 0
        for prediction_time in prediction_times:
            past_times = [
                prediction_time + i*prediction_interval for i in range(-past_timesteps + 1, 1)
            ]
            future_times = [
                prediction_time + i*prediction_interval for i in range(1, future_timesteps + 1)
            ]
            training_sets[row] = \
                [queue(time) for time in past_times for queue in flow.queues] + \
                [load(time) for time in past_times for load in edgeLoads] + \
                [queue(time) for time in future_times for queue in flow.queues]
        np.save(out_path, training_sets)


def expanded_queues_from_flows(network_path: str, past_timesteps: int, step_length: float, future_timesteps: int,
                               flows_folder: str, out_folder: str, horizon: int, sample_step: int):
    os.makedirs(out_folder, exist_ok=True)
    network = Network.from_file(network_path)
    out_path = os.path.join(out_folder, "expanded_queues.csv.gz")
    lock_path = os.path.join(out_folder, ".lock.expanded_queues.csv.gz")
    if os.path.exists(out_path):
        print("Expanded Queues were already computed. Skipping...")
        return
    elif os.path.exists(lock_path):
        print("Detected a lock file for expanded queues. Skipping...")
        return
    with open(lock_path, "w") as file:
        file.write("")

    files = [file for file in os.listdir(
        flows_folder) if file.endswith(".flow.pickle")]

    sample_times = range(
        0, floor(horizon - step_length * future_timesteps), sample_step)
    samples = []
    for flow_path in files:
        with open(os.path.join(flows_folder, flow_path), "rb") as file:
            flow = pickle.load(file)
        flow.network = network

        for phi in sample_times:
            for edge in network.graph.edges:
                d = {}
                for k, ie in enumerate(edge.node_from.incoming_edges):
                    for t in range(-past_timesteps, 1):
                        d[f"i{k}[{t}]"] = flow.queues[ie.id](
                            phi - step_length * t)
                for t in range(-past_timesteps, future_timesteps + 1):
                    d[f"e[{t}]"] = flow.queues[edge.id](phi - step_length * t)
                if any(v != 0. for v in d.values()):
                    samples.append(d)
    df = pd.DataFrame(samples, columns=[f"i{e}[{t}]" for e in range(5) for t in range(-past_timesteps, 1)]
                      + [f"e[{t}]" for t in range(-past_timesteps, future_timesteps + 1)])
    df.to_csv(out_path)
    print("Successfully written expanded queues to disk.")
    os.remove(lock_path)


def expanded_queues_from_flows_per_edge(network_path: str, past_timesteps: int, step_length: float,
                                        future_timesteps: int, flows_folder: str, out_folder: str, horizon: int,
                                        sample_step: int, average: bool):
    os.makedirs(out_folder, exist_ok=True)
    network = Network.from_file(network_path)
    flows = None

    for e in network.graph.edges:
        out_path = os.path.join(out_folder, f"edge-{e.id}.csv.gz")
        lock_path = os.path.join(out_folder, f".lock.edge-{e.id}.csv.gz")
        if os.path.exists(out_path):
            print(
                f"Expanded Queues for edge {e.id} were already computed. Skipping...")
            continue
        elif os.path.exists(lock_path):
            print(
                f"Detected a lock file for expanded queues for edge {e.id}. Skipping...")
            continue
        with open(lock_path, "w") as file:
            file.write("")

        if flows is None:
            flows = []
            flow_files = [file for file in os.listdir(flows_folder)
                          if file.endswith(".flow.pickle") and not file.startswith(".lock")]
            for flow_path in flow_files:
                with open(os.path.join(flows_folder, flow_path), "rb") as file:
                    flow: DynamicFlow = pickle.load(file)
                flow.network = network
                flows.append(flow)

        sample_times = range(
            0, floor(horizon - step_length * future_timesteps), sample_step)
        samples = []
        for flow in flows:
            for phi in sample_times:
                d = {}
                for ie in e.node_from.incoming_edges:
                    for t in range(-past_timesteps + 1, 1):
                        d[f"{ie.id}[{t}]"] = flow.queues[ie.id].integrate(phi - step_length * (t + 1),
                                                                          phi - step_length * t) \
                            if average else \
                            flow.queues[ie.id](phi - step_length * t)
                for oe in e.node_to.outgoing_edges:
                    for t in range(-past_timesteps + 1, 1):
                        d[f"{oe.id}[{t}]"] = flow.queues[oe.id].integrate(phi - step_length * (t + 1),
                                                                          phi - step_length * t) \
                            if average else \
                            flow.queues[oe.id](phi - step_length * t)
                for t in range(-past_timesteps + 1, future_timesteps + 1):
                    d[f"{e.id}[{t}]"] = flow.queues[e.id].integrate(phi - step_length * (t + 1),
                                                                    phi - step_length * t) \
                        if average else \
                        flow.queues[e.id](phi - step_length * t)
                samples.append(d)
        df = pd.DataFrame(samples, columns=samples[0].keys())
        df.to_csv(out_path)
        print(f"Successfully written expanded queues for edge {e.id} to disk.")
        os.remove(lock_path)
