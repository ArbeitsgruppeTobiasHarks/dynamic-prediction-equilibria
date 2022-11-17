import os
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import T_co, Dataset

from core.dynamic_flow import DynamicFlow
from core.network import Network


class QueueDataset(Dataset):
    _queues: List[Tensor]
    _flow: Optional[DynamicFlow]
    test_mask: torch.Tensor
    samples_per_flow: int

    def __init__(self, folder_path: str, past_timesteps: int, future_timesteps: int, network: Network, in_memory: bool,
                 torch_mode: str):
        self._in_memory = in_memory
        self._past_timesteps = past_timesteps
        self._future_timesteps = future_timesteps
        self._queue_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self._queues = []
        self._torch_mode = torch_mode
        assert len(self._queue_files) > 0
        if self._in_memory:
            print("Reading in dataset...")
            max_queues = torch.zeros(len(network.graph.edges))
            for file in self._queue_files:
                queues = torch.from_numpy(np.genfromtxt(file)).float()
                max_queues = torch.maximum(max_queues, torch.amax(queues, dim=1))
                self._queues.append(queues)
                self.samples_per_flow = queues.shape[1] - past_timesteps - future_timesteps
            print("Done reading dataset.")
            self.test_mask = torch.tensor([max_queue > 0 for max_queue in max_queues]).to(torch_mode)
            np.savetxt("../../out/mask.txt", self.test_mask)
        else:
            np_mask = np.genfromtxt("../../out/mask.txt")
            self.test_mask = torch.tensor([max_queue > 0 for max_queue in np_mask]).to(torch_mode)
            self.samples_per_flow = 20

    def __getitem__(self, index) -> T_co:
        flow_id, sample_id = divmod(index, self.samples_per_flow)
        if self._in_memory:
            queues = self._queues[flow_id]
        else:
            queues = torch.from_numpy(np.genfromtxt(self._queue_files[flow_id])).float()
        stride = (queues.shape[1] - self._past_timesteps - self._future_timesteps) // self.samples_per_flow
        phi = stride * sample_id + self._past_timesteps
        past_queues = queues[:, stride * sample_id: phi]
        future_queues = queues[:, phi: phi + self._future_timesteps]
        return past_queues.to(self._torch_mode)[self.test_mask], future_queues.to(self._torch_mode)[self.test_mask]

    def __len__(self):
        return (len(self._queues) if self._in_memory else len(self._queue_files)) * self.samples_per_flow
