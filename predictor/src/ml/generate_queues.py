import os
import pickle

import numpy as np
import pandas as pd
from math import floor

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network


def generate_queues(past_timesteps: int, flows_folder: str, out_folder: str, horizon: int, step_length: int):
    os.makedirs(out_folder, exist_ok=True)
    files = [file for file in os.listdir(flows_folder) if
             file.endswith(".flow.pickle") and not file.startswith(".lock.")]

    for flow_path in files:
        flow_id = flow_path[: len(flow_path) - len(".flow.pickle")]
        queue_path = os.path.join(out_folder, f"{flow_id}.csv.gz")
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

    files = [file for file in os.listdir(flows_folder) if file.endswith(".flow.pickle")]

    sample_times = range(0, floor(horizon - step_length * future_timesteps), sample_step)
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
                        d[f"i{k}[{t}]"] = flow.queues[ie.id](phi - step_length * t)
                for t in range(-past_timesteps, future_timesteps + 1):
                    d[f"e[{t}]"] = flow.queues[edge.id](phi - step_length * t)
                samples.append(d)
    df = pd.DataFrame(samples, columns=
    [f"i{e}[{t}]" for e in range(5) for t in range(-past_timesteps, 1)]
    + [f"e[{t}]" for t in range(-past_timesteps, future_timesteps + 1)])
    df.to_csv(out_path)
    print("Successfully written expanded queues to disk.")
    os.remove(lock_path)


def expanded_queues_from_flows_per_edge(network_path: str, past_timesteps: int, step_length: float,
                                        future_timesteps: int, flows_folder: str, out_folder: str, horizon: int,
                                        sample_step: int):
    os.makedirs(out_folder, exist_ok=True)
    network = Network.from_file(network_path)
    flows = None

    for e in network.graph.edges:
        out_path = os.path.join(out_folder, f"edge-{e.id}.csv.gz")
        lock_path = os.path.join(out_folder, f".lock.edge-{e.id}.csv.gz")
        if os.path.exists(out_path):
            print(f"Expanded Queues for edge {e.id} were already computed. Skipping...")
            continue
        elif os.path.exists(lock_path):
            print(f"Detected a lock file for expanded queues for edge {e.id}. Skipping...")
            continue
        with open(lock_path, "w") as file:
            file.write("")

        if flows is None:
            flows = []
            flow_files = [file for file in os.listdir(flows_folder) if file.endswith(".flow.pickle")]
            for flow_path in flow_files:
                with open(os.path.join(flows_folder, flow_path), "rb") as file:
                    flow: MultiComPartialDynamicFlow = pickle.load(file)
                flow.network = network
                flows.append(flow)

        sample_times = range(0, floor(horizon - step_length * future_timesteps), sample_step)
        samples = []
        for flow in flows:
            for phi in sample_times:
                d = {}
                for ie in e.node_from.incoming_edges:
                    for t in range(-past_timesteps + 1, 1):
                        d[f"{ie.id}[{t}]"] = flow.queues[ie.id].integrate(phi - step_length * (t + 1),
                                                                          phi - step_length * t)
                for oe in e.node_to.outgoing_edges:
                    for t in range(-past_timesteps + 1, 1):
                        d[f"{oe.id}[{t}]"] = flow.queues[oe.id].integrate(phi - step_length * (t + 1),
                                                                          phi - step_length * t)
                for t in range(-past_timesteps + 1, future_timesteps + 1):
                    d[f"{e.id}[{t}]"] = flow.queues[e.id].integrate(phi - step_length * (t + 1),
                                                                    phi - step_length * t)
                samples.append(d)
        df = pd.DataFrame(samples, columns=samples[0].keys())
        df.to_csv(out_path)
        print(f"Successfully written expanded queues for edge {e.id} to disk.")
        os.remove(lock_path)
