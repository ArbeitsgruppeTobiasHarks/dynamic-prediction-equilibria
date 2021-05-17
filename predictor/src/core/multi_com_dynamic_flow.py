from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from core.machine_precision import eps as machine_eps
from core.network import Network
from utilities.queues import PriorityQueue, PriorityItem


class MultiComPartialDynamicFlow:
    """
        This is a representation of a flow with right-constant edge inflow rates on intervals.
    """

    # We use List for fast extensions
    times: List[float]
    inflow: List[np.ndarray]  # inflow[k][i,e] is the constant value of fᵢₑ⁺ on [times[k], times[k+1])
    curr_outflow: np.ndarray  # outflow[i,e] is the constant value of fᵢₑ⁻ starting from times[-1]
    queues: List[np.ndarray]  # queue[k][e] is the queue length at e at time times[k]
    change_events: PriorityQueue[OutflowChangeEvent]  # A priority queue with times where some edge outflow changes

    _network: Network

    def __init__(self, network: Network):
        self._network = network
        n = len(network.commodities)
        m = len(network.graph.edges)
        self.times = [0.]
        self.inflow = []
        self.queues = [np.zeros(len(network.graph.edges))]
        self.curr_outflow = np.zeros((n, m))
        self.change_events = PriorityQueue()

    def extend(self, new_inflow: np.ndarray, max_extension_length: float, stop_at_queue_changes: bool) -> float:
        """
        Extends the flow with constant inflows new_inflow until some edge outflow changes.
        The user can also specify a maximum extension length using max_extension_length.
        If verify_balance is given, it checks if the extension satisfies the balances given.
        :returns The length of the actual extension
        """
        phi = self.times[-1]
        n = len(self._network.commodities)
        m = len(self._network.graph.edges)
        capacity = self._network.capacity
        travel_time = self._network.travel_time

        # determine how long we can extend without changing any edge outflow
        queue_depletion_events: List[QueueDepletionEvent] = []
        accum_inflow = np.sum(new_inflow, axis=0)
        for e in range(m):
            accum_edge_inflow = accum_inflow[e]
            if self.queues[-1][e] > 0:
                if accum_edge_inflow < capacity[e]:
                    depletion_time = phi + self.queues[-1][e] / (capacity[e] - accum_edge_inflow)
                    queue_depletion_events.append(QueueDepletionEvent(
                        edge=e, change_time=depletion_time + travel_time[e], depletion_time=depletion_time,
                        new_outflow=new_inflow[:, e]
                    ))
            if not np.array_equal(new_inflow[:, e], np.zeros(n) if phi == 0 else self.inflow[-1][:, e]):
                accum_outflow = min(capacity[e], accum_edge_inflow)
                new_outflow = np.zeros((n, 1)) if accum_outflow == 0 else \
                    accum_outflow * new_inflow[:, e] / accum_edge_inflow
                event = OutflowChangeEvent(e, phi + self.queues[-1][e] / capacity[e] + travel_time[e], new_outflow)
                self.change_events.push(PriorityItem(event.time, event))

        # Remove fake change_events
        while len(self.change_events) > 0:
            next_event = self.change_events.next()
            if np.array_equal(next_event.new_outflow, self.curr_outflow[:, next_event.edge]):
                self.change_events.pop()
            else:
                break

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
                    OutflowChangeEvent(depl_ev.edge, depl_ev.change_time, depl_ev.new_outflow),
                ))

        while self.change_events.min_time() <= new_phi + machine_eps:
            event = self.change_events.pop()
            self.curr_outflow[:, event.edge] = event.new_outflow.ravel()

        self.inflow.append(new_inflow)
        # qₑ(θ + ε) = max{ 0, qₑ(θ) + ε(fₑ⁺ - νₑ) }
        self.queues.append(np.maximum(0, self.queues[-1] + eps * (accum_inflow - capacity)))
        self.times.append(new_phi)

        return new_phi


@dataclass
class OutflowChangeEvent:
    edge: int
    time: float
    new_outflow: np.ndarray


@dataclass
class QueueDepletionEvent:
    edge: int
    change_time: float
    depletion_time: float
    new_outflow: np.ndarray
