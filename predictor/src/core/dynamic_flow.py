from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from core.network import Network
from utilities.queues import PriorityQueue, PriorityItem


class PartialDynamicFlow:
    """
        This is a representation of a flow with right-constant edge inflow rates on intervals.
    """

    # We use List for fast extensions
    times: List[float]
    inflow: List[np.ndarray]  # inflow[k][e] is the constant value of fₑ⁺ on [times[k], times[k+1])
    curr_outflow: np.ndarray  # outflow[e] is the constant value of fₑ⁻ starting from times[-1]
    queues: List[np.ndarray]  # queue[k][e] is the queue length at e at time times[k]
    change_events: PriorityQueue[OutflowChangeEvent]  # A priority queue with times where some edge outflow changes

    _network: Network

    def __init__(self, network: Network):
        self._network = network
        self.times = [0.]
        self.inflow = []
        self.queues = [np.zeros(len(network.graph.edges))]
        self.curr_outflow = np.zeros(len(network.graph.edges))
        self.change_events = PriorityQueue()

    def extend(self, new_inflow: np.ndarray, max_extension_length: float, stop_at_queue_changes: bool) -> float:
        """
        Extends the flow with constant inflows new_inflow until some edge outflow changes.
        The user can also specify a maximum extension length using max_extension_length.
        If verify_balance is given, it checks if the extension satisfies the balances given.
        :returns The length of the actual extension
        """
        phi = self.times[-1]
        m = len(self._network.graph.edges)
        capacity = self._network.capacity
        travel_time = self._network.travel_time

        # determine how long we can extend without changing any edge outflow
        queue_depletion_events: List[QueueDepletionEvent] = []
        for e in range(m):
            if self.queues[-1][e] > 0:
                if new_inflow[e] < capacity[e]:
                    depletion_time = phi + self.queues[-1][e] / (capacity[e] - new_inflow[e])
                    queue_depletion_events.append(QueueDepletionEvent(
                        edge=e, change_time=depletion_time + travel_time[e], depletion_time=depletion_time
                    ))
            elif self.queues[-1][e] == 0 and new_inflow[e] != (0 if phi == 0 else self.inflow[-1][e]):
                event = OutflowChangeEvent(e, phi + travel_time[e], new_outflow=min(capacity[e], new_inflow[e]))
                self.change_events.push(PriorityItem(event.time, event))

        first_change_time = min(
            min((event.change_time for event in queue_depletion_events), default=float('inf')),
            self.change_events.min_time()
        )

        # Finally: The actual extension length
        eps = min(
            first_change_time - phi,
            max_extension_length,
            min((event.depletion_time for event in queue_depletion_events), default=float('inf'))
            if stop_at_queue_changes else
            float('inf')
        )
        new_phi = phi + eps
        for depl_ev in queue_depletion_events:
            if depl_ev.depletion_time <= new_phi:
                self.change_events.push(PriorityItem(
                    depl_ev.change_time,
                    OutflowChangeEvent(depl_ev.edge, depl_ev.change_time, new_inflow[depl_ev.edge])
                ))

        while self.change_events.min_time() <= new_phi:
            event = self.change_events.pop()
            self.curr_outflow[event.edge] = event.new_outflow

        self.inflow.append(new_inflow)
        # qₑ(θ + ε) = max{ 0, qₑ(θ) + ε(fₑ⁺ - νₑ) }
        self.queues.append(np.maximum(0, self.queues[-1] + eps * (new_inflow - capacity)))
        self.times.append(new_phi)

        return new_phi

    def verify_balance(self, new_inflow: np.ndarray, verify_balance: Dict[int, float]):
        for (node_id, balance) in verify_balance:
            assert node_id in self._network.graph.nodes.keys(), f"Could not find node#{node_id}"
            node = self._network.graph.nodes[node_id]
            node_inflow = sum(self.curr_outflow[e.id] for e in node.incoming_edges)
            node_outflow = sum(new_inflow[e.id] for e in node.outgoing_edges)
            assert balance == node_inflow - node_outflow, f"Balance for node#{node_id} does not match"


@dataclass
class OutflowChangeEvent:
    edge: int
    time: float
    new_outflow: float


@dataclass
class QueueDepletionEvent:
    edge: int
    change_time: float
    depletion_time: float
