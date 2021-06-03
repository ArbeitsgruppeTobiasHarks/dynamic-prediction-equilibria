import os
import pickle
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import T_co, Dataset

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network


class QueueDataset(Dataset):
    _queues: List[Tuple[Tensor, Tensor]]
    _flow: Optional[MultiComPartialDynamicFlow]
    test_mask: torch.Tensor

    def __init__(self, folder_path: str, past_timesteps: int, future_timesteps: int, network: Network,
                 in_memory: bool):
        self._in_memory = in_memory
        self._future_timesteps = future_timesteps
        self._queue_dirs = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self._queues = []
        assert len(self._queue_dirs) > 0
        if self._in_memory:
            print("Reading in dataset...")
            max_queues = torch.zeros(len(network.graph.edges))
            for directory in self._queue_dirs:
                past_queues = torch.load(os.path.join(directory, "past_queues.pt"))
                future_queues = torch.load(os.path.join(directory, "future_queues.pt"))[:, 0:future_timesteps]
                max_queues = torch.maximum(max_queues,
                                           torch.maximum(
                                               torch.amax(past_queues, dim=1),
                                               torch.amax(future_queues, dim=1)
                                           ))
                self._queues.append((past_queues, future_queues))
            print("Done reading dataset.")
            self.test_mask = torch.tensor([max_queue > 0 for max_queue in max_queues]).to('cuda')
        else:
            np_mask = np.genfromtxt("../../out/mask.txt")
            self.test_mask = torch.tensor([max_queue > 0 for max_queue in np_mask]).to('cuda')

    def __getitem__(self, index) -> T_co:
        if self._in_memory:
            past_queues = self._queues[index][0]
            future_queues = self._queues[index][1]
        else:
            past_queues = torch.load(os.path.join(self._queue_dirs[index], "past_queues.pt"))
            future_queues = torch.load(
                os.path.join(self._queue_dirs[index], "future_queues.pt")
            )[:, 0:self._future_timesteps]
        return past_queues.to('cuda')[self.test_mask], future_queues.to('cuda')[self.test_mask]

    def __len__(self):
        return len(self._queues) if self._in_memory else len(self._queue_dirs)


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
            past_queues = torch.tensor([
                [queue(time) for time in times_past_queues] for queue in flow.queues
            ], dtype=torch.float32)
            future_queues = torch.tensor([
                [queue(time) for time in times_future_queues] for queue in flow.queues
            ], dtype=torch.float32)

            os.makedirs(os.path.join(out_folder, f"{flow_path}.{i}"), exist_ok=True)

            with open(os.path.join(out_folder, f"{flow_path}.{i}/past_queues.pt"), "wb") as file:
                torch.save(past_queues, file)

            with open(os.path.join(out_folder, f"{flow_path}.{i}/future_queues.pt"), "wb") as file:
                torch.save(future_queues, file)


if __name__ == '__main__':
    past_timesteps, future_timesteps = 5, 5
    flows_folder = "/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/generated_flows/"
    output_folder = f"../../out/generated_queues/{past_timesteps},{future_timesteps}/"
    generate_queues(5, 5, flows_folder, output_folder)
