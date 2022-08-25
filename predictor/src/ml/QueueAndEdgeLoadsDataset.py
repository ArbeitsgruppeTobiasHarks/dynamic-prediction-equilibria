from math import floor
import os
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data.dataset import T_co, Dataset

from core.dynamic_flow import DynamicFlow
from core.network import Network


class QueueAndEdgeLoadDataset(Dataset):
    _data: np.ndarray  # Dim0: Flow, Dim1: Queues/EdgeLoad, Dim2: Edge, Dim3: Time
    _flow: Optional[DynamicFlow]
    _reroute_interval: float
    _prediction_interval: float
    _times: List[float]
    input_mask: np.ndarray
    output_mask: np.ndarray

    def __init__(self, folder_path: str, past_timesteps: int, future_timesteps: int,  reroute_interval: float, prediction_interval: float, horizon: float, network: Network):
        self._past_timesteps = past_timesteps
        self._future_timesteps = future_timesteps
        self._data_files = [os.path.join(folder_path, file)
                            for file in sorted(os.listdir(folder_path))]
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
        self._additional_input_mask = None
        self._additional_output_mask = None

        if len(self._data_files) == 0:
            raise ValueError("There are no queue files.")

        print("Reading in dataset...")
        max_data = np.zeros(len(network.graph.edges))
        max_queue = np.zeros(len(network.graph.edges))
        all_data = []
        for file in self._data_files:
            data = np.load(file)
            if data.shape != (2, len(network.graph.edges), len(self._times)):
                raise ValueError(
                    f"Queue data has wrong shape: Expected (2, {len(network.graph.edges)}, {len(self._times)}), but got {data.shape}.")
            max_data = np.maximum(max_data, np.amax(data, axis=(0, 2)))
            max_queue = np.maximum(max_queue, np.amax(data[0], axis=(1)))
            all_data.append(data)
            self.samples_per_flow = data.shape[2] - self._reroute_intervals_in_prediction_interval * (
                past_timesteps + future_timesteps)
        self._data = np.asarray(all_data)
        print("Done reading dataset.")
        self.input_mask = np.asarray([val > 0 for val in max_data])
        self.output_mask = np.asarray([val > 0 for val in max_queue])
        np.savetxt(os.path.join(
            folder_path, "../input-mask.txt"), self.input_mask)
        np.savetxt(os.path.join(folder_path, "../output-mask.txt"),
                   self.output_mask)

    def use_additional_input_mask(self, additional_input_mask):
        self._additional_input_mask = additional_input_mask
    def use_additional_output_mask(self, additional_output_mask):
        self._additional_output_mask = additional_output_mask

    def __getitem__(self, index) -> T_co:
        flow_id, sample_id = divmod(index, self.samples_per_flow)
        data = self._data[flow_id]
        phi = sample_id * self._reroute_interval
        first_ind = sample_id
        stride = self._reroute_intervals_in_prediction_interval
        phi_ind = sample_id + (self._past_timesteps - 1) * stride
        last_ind = phi_ind + self._future_timesteps * stride
        input_mask = self.input_mask if self._additional_input_mask is None else self.input_mask * self._additional_input_mask
        output_mask = self.output_mask if self._additional_output_mask is None else self.output_mask * self._additional_output_mask

        past_data = np.array(
            [phi, *(data[:, input_mask, first_ind: phi_ind + 1: stride].flatten())])
        future_queues = data[0, output_mask,
                             phi_ind + stride: last_ind + 1: stride].flatten()
        return past_data, future_queues

    def __len__(self):
        return len(self._data) * self.samples_per_flow

    @staticmethod
    def load_mask(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        input_mask = np.array([v > 0 for v in np.loadtxt(
            os.path.join(folder_path, "../input-mask.txt"))])
        output_mask = np.array([v > 0 for v in np.loadtxt(
            os.path.join(folder_path, "../output-mask.txt"))])
        return input_mask, output_mask

    @staticmethod
    def mask_exists(folder_path: str) -> np.ndarray:
        return os.path.exists(os.path.join(folder_path, "../input-mask.txt")) \
            and os.path.exists(os.path.join(folder_path, "../output-mask.txt"))
