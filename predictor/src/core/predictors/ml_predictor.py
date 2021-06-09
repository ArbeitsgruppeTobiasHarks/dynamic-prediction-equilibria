from __future__ import annotations

from typing import List

import dgl
import numpy as np
import torch
from dgl._deprecate.graph import DGLGraph

from core.network import Network
from core.predictor import Predictor, PredictionResult
from gnn.Model import Model
from utilities.interpolate import LinearlyInterpolatedFunction


class MLPredictor(Predictor):
    _past_timesteps: int
    _future_timesteps: int
    _step_size: float
    _model: Model
    _capacity: torch.Tensor
    _travel_time: torch.Tensor
    _dgl_graph: DGLGraph

    def __init__(self, network: Network, model_checkpoint_path: str):
        super().__init__(network)
        self._past_timesteps, self._future_timesteps, self._step_size = 5, 5, 1.0
        self._model = Model(self._past_timesteps, self._future_timesteps, 20)
        checkpoint = torch.load(model_checkpoint_path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        self._capacity = torch.from_numpy(network.capacity).float()
        self._travel_time = torch.from_numpy(network.travel_time).float()

        new_edges = [(e1.id, e2.id) for e1 in network.graph.edges for e2 in e1.node_to.outgoing_edges]
        u = torch.tensor([e[0] for e in new_edges])
        v = torch.tensor([e[1] for e in new_edges])
        self._dgl_graph = dgl.graph((u, v)).add_self_loop()

    def predict_from_fcts(self, old_queues: List[LinearlyInterpolatedFunction], phi: float) -> PredictionResult:
        times_past_queues = [
            phi - i * i * self._step_size for i in range(self._past_timesteps)
        ]
        times_future_queues = [
            phi + i * i * self._step_size for i in range(self._future_timesteps + 1)
        ]
        times_future_queues.append(times_future_queues[-1] + 1)
        past_queues = torch.tensor([
            [queue(time) for time in times_past_queues] for queue in old_queues
        ], dtype=torch.float32)
        input_tensor = torch.column_stack([self._capacity, self._travel_time, past_queues])

        future_queues: torch.Tensor = self._model(self._dgl_graph, input_tensor)
        queues: List[np.ndarray] = [np.array([old_queue(phi) for old_queue in old_queues])] + \
                                   [future_queues[:, i].numpy() for i in range(self._future_timesteps)] + \
                                   [future_queues[:, -1].numpy()]
        return PredictionResult(times_future_queues, queues)

    def type(self) -> str:
        return "ML Predictor"

    def predict(self, times: List[float], old_queues: List[np.ndarray]) -> PredictionResult:
        raise NotImplementedError()
