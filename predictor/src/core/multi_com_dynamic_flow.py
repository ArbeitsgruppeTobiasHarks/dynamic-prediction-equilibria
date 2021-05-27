from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

import numpy as np

from core.machine_precision import eps as machine_eps
from core.network import Network
from utilities.interpolate import LinearlyInterpolatedFunction
from utilities.queues import PriorityQueue
from utilities.right_constant import RightConstantFunction


class MultiComPartialDynamicFlow:
    """
        This is a representation of a flow with right-constant edge inflow rates on intervals.
    """

    # We use List for fast extensions
    phi: float
    inflow: List[List[RightConstantFunction]]  # inflow[e][i] is the function fᵢₑ⁺
    outflow: List[List[RightConstantFunction]]  # outflow[e][i] is the function fᵢₑ⁻
    queues: List[LinearlyInterpolatedFunction]  # queue[e] is the queue length at e
    outflow_changes: PriorityQueue[Tuple[int, float]]  # A priority queue with times where some edge outflow changes
    queue_depletions: PriorityQueue[int]  # A priority queue with times where queues deplete
    depletion_dict: Dict[int, float]  # Maps an edge with ongoing depletion_to outflow change time and new_outflow

    _network: Network

    def __init__(self, network: Network):
        self._network = network
        n = len(network.commodities)
        m = len(network.graph.edges)
        self.phi = 0.
        self.inflow = [[RightConstantFunction([self.phi], [0.], (self.phi, float('inf'))) for i in range(n)] for e in
                       range(m)]
        self.queues = [LinearlyInterpolatedFunction([self.phi - 1, self.phi], [0., 0.]) for e in range(m)]
        self.outflow = [[RightConstantFunction([self.phi], [0.], (self.phi, float('inf'))) for i in range(n)] for e in
                        range(m)]
        self.outflow_changes = PriorityQueue()
        self.queue_depletions = PriorityQueue()
        self.depletion_dict = {}

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Don't pickle _network b.c. of recursive structure
        del state["_network"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        print("Please reset network on flow before accessing its functions")

    def extend(self, new_inflow: Dict[int, np.ndarray], max_extension_length: float) -> Set[int]:
        """
        Extends the flow with constant inflows new_inflow until some edge outflow changes.
        Edge inflows not in new_inflow are extended with their previous values.
        The user can also specify a maximum extension length using max_extension_length.
        :returns set of edges where the outflow has changed
        """
        phi = self.phi
        n = len(self._network.commodities)
        capacity = self._network.capacity
        travel_time = self._network.travel_time
        queue_updates_when_eps_known = []

        for e in new_inflow.keys():

            accum_edge_inflow = sum(new_inflow[e])
            cur_queue = self.queues[e](phi)
            #assert cur_queue >= 0
            cur_queue = max(0., cur_queue)

            # UPDATE QUEUE_DEPLETIONS
            if cur_queue > 0:
                if accum_edge_inflow < capacity[e]:
                    depletion_time = phi + cur_queue / (capacity[e] - accum_edge_inflow)
                    if self.queue_depletions.has(e):
                        self.queue_depletions.update(e, depletion_time)
                        self.depletion_dict[e] = depletion_time + travel_time[e]
                    else:
                        self.queue_depletions.push(e, depletion_time)
                        self.depletion_dict[e] = depletion_time + travel_time[e]
                elif self.queue_depletions.has(e):
                    self.queue_depletions.remove(e)
                    self.depletion_dict.pop(e)

            # UPDATE OUTFLOW, OUTFLOW CHANGE_EVENTS AND QUEUES
            if any(new_inflow[e][i] != self.inflow[e][i].values[-1] for i in range(n)):
                accum_outflow = min(capacity[e], accum_edge_inflow)
                change_time = phi + cur_queue / capacity[e] + travel_time[e]
                new_outflow = np.zeros(n) if accum_outflow == 0 else accum_outflow * new_inflow[e] / accum_edge_inflow
                for i in range(n):
                    self.outflow[e][i].extend(change_time, new_outflow[i])
                if not self.outflow_changes.has((e, change_time)):
                    self.outflow_changes.push((e, change_time), change_time)
                self.queues[e].extend(phi, cur_queue)
                queue_updates_when_eps_known.append((e, cur_queue, accum_edge_inflow - capacity[e]))

            # EXTEND INFLOW
            for i in range(n):
                self.inflow[e][i].extend(phi, new_inflow[e][i])

        first_change_time = self.outflow_changes.min_key()
        # Finally: The actual extension length
        eps = min(
            first_change_time - phi,
            max_extension_length
        )
        new_phi = phi + eps

        # REFLECT QUEUE CHANGES DUE TO DEPLETIONS
        add_again = []
        while self.queue_depletions.min_key() <= new_phi + machine_eps:
            depletion_time = self.queue_depletions.min_key()
            e = self.queue_depletions.pop()
            if self.queues[e].times[-1] <= depletion_time + machine_eps:
                self.queues[e].extend(depletion_time, 0.)
            if depletion_time <= new_phi - machine_eps:
                self.queues[e].extend(new_phi, 0.)
                self.depletion_dict.pop(e)
            else:
                add_again.append((e, depletion_time))
        for e, depletion_time in add_again:
            self.queue_depletions.push(e, depletion_time)

        # REFLECT QUEUE CHANGES DUE TO INFLOW CHANGE
        for (e, cur_queue, slope) in queue_updates_when_eps_known:
            self.queues[e].extend(new_phi, max(0., cur_queue + eps * slope))

        changed_edges: Set[int] = set()
        while self.outflow_changes.min_key() <= new_phi + machine_eps:
            (e, _) = self.outflow_changes.pop()
            changed_edges.add(e)

        self.phi = new_phi
        return changed_edges

    def avg_travel_time(self, i: int, horizon: float) -> float:
        commodity = self._network.commodities[i]
        net_outflow: RightConstantFunction = sum(self.outflow[e.id][i] for e in commodity.sink.incoming_edges)
        accum_net_outflow = net_outflow.integral()
        avg_travel_time = horizon / 2 - accum_net_outflow.integrate(0., horizon) / (horizon * commodity.demand)
        return avg_travel_time


class OutflowChangeEvent:
    edge: int
    time: float
    new_outflow: np.ndarray

    def __init__(self, edge: int, time: float, new_outflow: np.ndarray):
        self.edge = edge
        self.time = time
        self.new_outflow = new_outflow


@dataclass
class QueueDepletionEvent:
    edge: int
    change_time: float
    depletion_time: float
    new_outflow: np.ndarray
