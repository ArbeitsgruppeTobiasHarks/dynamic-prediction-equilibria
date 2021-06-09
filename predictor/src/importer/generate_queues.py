import os
import pickle
import random

import numpy as np


def generate_queues(past_timesteps: int, future_timesteps: int, flows_folder: str, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    samples_per_flow = 200
    step_size = 1.

    files = [file for file in os.listdir(flows_folder) if file.endswith(".flow.pickle")]

    for flow_path in files:
        with open(os.path.join(flows_folder, flow_path), "rb") as file:
            flow = pickle.load(file)

        for i in range(samples_per_flow):
            random.seed(i)
            pred_time = float(random.randint(0, 100))

            times_past_queues = [
                pred_time - i * step_size for i in range(past_timesteps)
            ]
            times_future_queues = [
                pred_time + i * step_size for i in range(1, future_timesteps + 1)
            ]
            times = times_past_queues + times_future_queues
            queues = np.asarray([
                [queue(time) for time in times] for queue in flow.queues
            ])
            np.savetxt(os.path.join(out_folder, f"{flow_path}.{i}.csv"), queues)
