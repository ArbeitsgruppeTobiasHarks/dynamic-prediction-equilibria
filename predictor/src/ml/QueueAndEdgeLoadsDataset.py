from math import floor
import os
from turtle import shape
from typing import List, Optional

import numpy as np
from torch.utils.data.dataset import T_co, Dataset

from core.dynamic_flow import DynamicFlow
from core.network import Network


class QueueAndEdgeLoadDataset(Dataset):
    _data: np.ndarray  # Dim0: Flow, Dim1: Queues/EdgeLoad, Dim2: Edge, Dim3: Time
    _flow: Optional[DynamicFlow]
    _reroute_interval: float
    _prediction_interval: float
    test_mask: np.ndarray
    _times: List[float]

    def __init__(self, folder_path: str, past_timesteps: int, future_timesteps: int,  reroute_interval: float, prediction_interval: float, horizon: float, network: Network):
        self._past_timesteps = past_timesteps
        self._future_timesteps = future_timesteps
        self._data_files = [os.path.join(folder_path, file)
                            for file in os.listdir(folder_path)]
        self._network = network
        self._reroute_interval = reroute_interval
        self._prediction_interval = prediction_interval
        self._reroute_intervals_in_prediction_interval = round(
            prediction_interval / reroute_interval)
        self._times = [
            i*reroute_interval
            for i in range(
                -(past_timesteps - 1) *
                self._reroute_intervals_in_prediction_interval,
                floor(horizon / reroute_interval) + 1
            )
        ]

        if len(self._data_files) == 0:
            raise ValueError("There are no queue files.")

        print("Reading in dataset...")
        max_data = np.zeros(len(network.graph.edges))
        all_data = []
        for file in self._data_files:
            data = np.load(file)
            if data.shape != (2, len(network.graph.edges), len(self._times)):
                raise ValueError(
                    f"Queue data has wrong shape: Expected (2, {len(network.graph.edges)}, {len(self._times)}), but got {data.shape}.")
            max_data = np.maximum(max_data, np.amax(data, axis=(0, 2)))
            all_data.append(data)
            self.samples_per_flow = data.shape[2] - self._reroute_intervals_in_prediction_interval * (
                past_timesteps + future_timesteps)
        self._data = np.asarray(all_data)
        print("Done reading dataset.")
        self.test_mask = np.asarray([val > 0 for val in max_data])
        np.savetxt(os.path.join(folder_path, "../mask.txt"), self.test_mask)

    def __getitem__(self, index) -> T_co:
        flow_id, sample_id = divmod(index, self.samples_per_flow)
        data = self._data[flow_id]
        phi = sample_id * self._reroute_interval
        first_ind = sample_id
        stride = self._reroute_intervals_in_prediction_interval
        phi_ind = sample_id + (self._past_timesteps - 1) * stride
        last_ind = phi_ind + self._future_timesteps * stride

        past_data = np.reshape(data[:, :, first_ind: phi_ind + 1: stride], newshape=(
            len(self._network.graph.edges), 2*self._past_timesteps))
        future_queues = (
            data[0, self.test_mask, phi_ind + stride: last_ind + 1: stride]).flatten()
        return np.concatenate(([phi], past_data[self.test_mask].flatten())), future_queues

    def __len__(self):
        return len(self._data) * self.samples_per_flow

    @staticmethod
    def load_mask(folder_path: str) -> np.ndarray:
        return np.array([v > 0 for v in np.loadtxt(os.path.join(folder_path, "../mask.txt"))])
