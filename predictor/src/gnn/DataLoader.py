import os
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
