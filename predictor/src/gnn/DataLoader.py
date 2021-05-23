import os
import pickle
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data.dataset import T_co, Dataset

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network


class QueueDataset(Dataset):
    _queues: List[Tuple[Tensor, Tensor]]
    _flow: Optional[MultiComPartialDynamicFlow]

    def __init__(self, folder_path: str, past_timesteps: int, future_timesteps: int, network: Network):
        queue_dirs = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self._queues = []
        assert len(queue_dirs) > 0
        print("Reading in dataset...")
        for directory in queue_dirs:
            past_queues = torch.load(os.path.join(directory, "past_queues.pt"))
            future_queues = torch.load(os.path.join(directory, "future_queues.pt"))
            self._queues.append((past_queues, future_queues))
        print("Done reading dataset.")

    def __getitem__(self, index) -> T_co:
        return self._queues[index][0].to('cuda'), self._queues[index][1].to('cuda')

    def __len__(self):
        return len(self._queues)


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
