from typing import List, Optional, Dict

import numpy as np

from core.network import Network


class PartialDynamicFlow:
    """
        This is a representation of a flow with constant edge inflow rates on discrete time steps of equal length.
        We assume, that all edge travel times are a multiple of the time_step
    """

    # We use List for fast extensions
    inflow: List[np.ndarray]  # inflow[t][e] is the constant inflow in e at times [t, t + time_step_size)
    queues: List[np.ndarray]  # queue[t][e] is the queue length at e at time t

    __time_step_size: int
    __network: Network

    def __init__(self, time_step_size: int, network: Network):
        self.__network = network
        self.__time_step_size = time_step_size
        assert all(tau % time_step_size == 0 for tau in network.travel_time)
        self.inflow = []
        self.queues = [np.zeros(len(network.graph.edges))]

    def extend(self, new_inflow: np.ndarray, verify_balance: Optional[Dict[int, float]]):
        """
        Extends the flow with constant inflows new_inflow until the next time step.
        If verify_balance is given, it checks if the extension satisfies the balances given
        """

        self.inflow.append(new_inflow)
        # qₑ(θ + ε) = max{ 0, qₑ(θ) + ε(fₑ⁺ - νₑ) }
        self.queues.append(np.maximum(
            0,
            self.queues[-1] + self.__time_step_size * (new_inflow - self.__network.capacity)
        ))

        if verify_balance is not None:
            current_outflow = self.current_outflow()
            for (node_id, balance) in verify_balance:
                assert node_id in self.__network.graph.nodes.keys(), f"Could not find node#{node_id}"
                node = self.__network.graph.nodes[node_id]
                node_inflow = sum(current_outflow[e.id] for e in node.incoming_edges)
                node_outflow = sum(new_inflow[e.id] for e in node.outgoing_edges)
                assert balance == node_inflow - node_outflow, f"Balance for node#{node_id} does not match"

    def current_outflow(self) -> np.ndarray:
        t = len(self.inflow)
        m = len(self.__network.graph.edges)
        tau = self.__network.travel_time

        queue_on_entry: np.ndarray = np.asarray([
            tau[e] > t or self.queues[t - tau[e]][1] > 0 for e in range(m)
        ])
        inflow_on_entry: np.ndarray = np.asarray([
            self.inflow[t - tau[e]] if tau[e] > t else 0 for e in range(m)
        ])

        return np.where(queue_on_entry, self.__network.capacity, inflow_on_entry)
