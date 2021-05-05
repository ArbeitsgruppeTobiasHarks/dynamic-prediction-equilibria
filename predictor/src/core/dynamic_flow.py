from heapq import heappush, heappop
from typing import List, Dict

import numpy as np

from core.network import Network


class PartialDynamicFlow:
    """
        This is a representation of a flow with right-constant edge inflow rates on intervals.
    """

    # We use List for fast extensions
    times: List[float]  #
    inflow: List[np.ndarray]  # inflow[k][e] is the constant value of fₑ⁺ on [times[k], times[k+1])
    queues: List[np.ndarray]  # queue[k][e] is the queue length at e at time times[k]
    change_events: List[float]  # A heapq priority queue with times where some edge outflow changes

    __network: Network

    def __init__(self, network: Network):
        self.__network = network
        self.times = [0.]
        self.inflow = []
        self.queues = [np.zeros(len(network.graph.edges))]
        self.change_events = []

    def extend(self, new_inflow: np.ndarray, max_extension_length: float) -> float:
        """
        Extends the flow with constant inflows new_inflow until some edge outflow changes.
        The user can also specify a maximum extension length using max_extension_length.
        If verify_balance is given, it checks if the extension satisfies the balances given.
        :returns The length of the actual extension
        """
        phi = self.times[-1]
        m = len(self.__network.graph.edges)
        capacity = self.__network.capacity
        travel_time = self.__network.travel_time

        # determine how long we can extend without changing any edge outflow
        for e in range(m):
            if self.queues[-1][e] > 0:
                if new_inflow[e] < capacity[e]:
                    heappush(
                        self.change_events,
                        phi + travel_time[e] + self.queues[-1][e] / (capacity[e] - new_inflow[e])
                    )
            elif self.queues[-1][e] == 0 and new_inflow[e] != (0 if phi == 0 else self.inflow[-1][e]):
                heappush(self.change_events, phi + travel_time[e])

        first_event = min(self.change_events, default=float('inf'))
        eps = min(first_event - phi, max_extension_length)
        if eps < max_extension_length:
            while min(self.change_events, default=float('inf')) <= first_event:
                heappop(self.change_events)

        self.inflow.append(new_inflow)
        # qₑ(θ + ε) = max{ 0, qₑ(θ) + ε(fₑ⁺ - νₑ) }
        self.queues.append(np.maximum(0, self.queues[-1] + eps * (new_inflow - capacity)))
        self.times.append(phi + eps)

        return phi + eps

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

    def verify_balance(self, new_inflow: np.ndarray, verify_balance: Dict[int, float]):
        current_outflow = self.current_outflow()
        for (node_id, balance) in verify_balance:
            assert node_id in self.__network.graph.nodes.keys(), f"Could not find node#{node_id}"
            node = self.__network.graph.nodes[node_id]
            node_inflow = sum(current_outflow[e.id] for e in node.incoming_edges)
            node_outflow = sum(new_inflow[e.id] for e in node.outgoing_edges)
            assert balance == node_inflow - node_outflow, f"Balance for node#{node_id} does not match"
