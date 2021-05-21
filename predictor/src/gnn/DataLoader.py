import os
import pickle
import random
from typing import List, Optional

import torch
from torch.utils.data.dataset import T_co, Dataset

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network


class QueueDataset(Dataset):
    _queue_dirs: List[str]
    _flow: Optional[MultiComPartialDynamicFlow]

    def __init__(self, folder_path: str, past_timesteps: int, future_timesteps: int, network: Network):
        self._queue_dirs = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        assert len(self._queue_dirs) > 0
        self.start = 0
        self.end = len(self._queue_dirs)
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps

        # Make everything ready for graph creation
        self._capacity = torch.from_numpy(network.capacity).float()
        self._travel_time = torch.from_numpy(network.travel_time).float()

        new_edges = [(e1.id, e2.id) for e1 in network.graph.edges for e2 in e1.node_to.outgoing_edges]
        self._u = torch.tensor([e[0] for e in new_edges])
        self._v = torch.tensor([e[1] for e in new_edges])

    def __getitem__(self, index) -> T_co:
        past_queues = torch.load(os.path.join(self._queue_dirs[index], "past_queues.pt"))
        future_queues = torch.load(os.path.join(self._queue_dirs[index], "future_queues.pt"))
        return (past_queues, future_queues)

    def __len__(self):
        return len(self._queue_dirs)


def generate_queues(past_timesteps: int, future_timesteps: int):
    flow_folder = "../../out/generated_flows/"
    output_folder = f"../../out/generated_queues/{past_timesteps},{future_timesteps}/"
    samples_per_flow = 20
    step_size = 1.

    files = [file for file in os.listdir(flow_folder) if file.endswith(".flow.pickle")]

    for flow_path in files:
        with open(os.path.join(flow_folder, flow_path), "rb") as file:
            flow = pickle.load(file)

        for i in range(samples_per_flow):
            random.seed(i)
            pred_time = float(random.randint(0, 300))
            times_past_queues = [
                pred_time - i * i * step_size for i in range(past_timesteps)
            ]
            times_future_queues = [
                pred_time + (i + 1) * (i + 1) * step_size for i in range(future_timesteps)
            ]
            past_queues = torch.tensor([
                [queue(time) for time in times_past_queues] for queue in flow.queues
            ], dtype=torch.float32)
            future_queues = torch.tensor([
                [queue(time) for time in times_future_queues] for queue in flow.queues
            ], dtype=torch.float32)

            os.makedirs(os.path.join(output_folder, f"{flow_path}.{i}"))

            with open(os.path.join(output_folder, f"{flow_path}.{i}/past_queues.pt"), "wb") as file:
                torch.save(past_queues, file)

            with open(os.path.join(output_folder, f"{flow_path}.{i}/future_queues.pt"), "wb") as file:
                torch.save(future_queues, file)


if __name__ == '__main__':
    generate_queues(5, 5)
