import os
import pickle

import numpy as np


def generate_queues(past_timesteps: int, future_timesteps: int, flows_folder: str, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    horizon = 100

    files = [file for file in os.listdir(flows_folder) if file.endswith(".flow.pickle")]

    for flow_path in files:
        with open(os.path.join(flows_folder, flow_path), "rb") as file:
            flow = pickle.load(file)

        times = range(-past_timesteps, horizon + future_timesteps)
        queues = np.asarray([
            [queue(time) for time in times] for queue in flow.queues
        ])
        np.savetxt(os.path.join(out_folder, f"{flow_path}.csv.gz"), queues)
