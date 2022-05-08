from __future__ import annotations

from typing import List, Dict, Set, Tuple, Optional

from core.machine_precision import eps
from core.network import Network
from utilities.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue
from utilities.right_constant import RightConstant


class DepletionQueue:
    depletions: PriorityQueue[int]
    change_times: PriorityQueue[int]
    new_outflow: Dict[int, List[float]]

    def __init__(self):
        self.depletions = PriorityQueue()
        self.change_times = PriorityQueue()
        self.new_outflow = {}

    def set(self, edge: int, depletion_time: float, change_event: Optional[Tuple[float, List[float]]] = None):
        assert depletion_time > float("-inf")
        self.depletions.set(edge, depletion_time)

        if change_event is not None:
            self.new_outflow[edge] = change_event[1]
            self.change_times.set(edge, change_event[0])
        elif edge in self.change_times:
            self.change_times.remove(edge)
            self.new_outflow.pop(edge)

    def __contains__(self, edge):
        return edge in self.depletions

    def remove(self, edge: int):
        self.depletions.remove(edge)
        if edge in self.change_times:
            self.change_times.remove(edge)
            self.new_outflow.pop(edge)

    def min_change_time(self):
        return self.change_times.min_key()

    def min_depletion(self):
        return self.depletions.min_key()

    def pop_by_depletion(self):
        depl_time, e = self.depletions.min_key(), self.depletions.pop()
        change_event = None
        if e in self.change_times:
            change_time = self.change_times.key_of(e)
            self.change_times.remove(e)
            new_outflow = self.new_outflow.pop(e)
            change_event = (change_time, new_outflow)
        return e, depl_time, change_event


class DynamicFlow:
    """
        This is a representation of a flow with right-constant edge inflow rates on intervals.
    """

    phi: float
    inflow: List[List[RightConstant]]  # inflow[e][i] is the function fᵢₑ⁺
    outflow: List[List[RightConstant]]  # outflow[e][i] is the function fᵢₑ⁻
    queues: List[PiecewiseLinear]  # queues[e] is the queue length at e
    outflow_changes: PriorityQueue[Tuple[int, float]]  # A priority queue with times where some edge outflow changes
    depletions: DepletionQueue  # A priority queue with events where queues deplete
    _network: Network

    def __init__(self, network: Network):
        self._network = network
        n = len(network.commodities)
        m = len(network.graph.edges)
        self.phi = 0.
        self.inflow = [[RightConstant([self.phi], [0.], (self.phi, float('inf'))) for _ in range(n)] for _ in
                       range(m)]
        self.queues = [PiecewiseLinear([self.phi], [0.], 0., 0.) for _ in range(m)]
        self.outflow = [[RightConstant([self.phi], [0.], (self.phi, float('inf'))) for _ in range(n)] for _ in
                        range(m)]
        self.outflow_changes = PriorityQueue()
        self.depletions = DepletionQueue()

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Don't pickle _network b.c. of recursive structure
        del state["_network"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        print("Please reset network on flow before accessing its functions")

    def _update_queue_depletions(self, phi: float, edge: int, accum_edge_inflow: float, cur_queue: float):
        """
            Update the queue_depletions queue and its corresponding dictionary depletion_dict for the change times.
        """
        capacity = self._network.capacity
        travel_time = self._network.travel_time
        if cur_queue > 0:
            if accum_edge_inflow < capacity[edge]:
                depletion_time = phi + cur_queue / (capacity[edge] - accum_edge_inflow)
                assert self.queues[edge](depletion_time) < 1000 * eps
                self.depletions.set(edge, depletion_time, depletion_time + travel_time[edge])
            elif edge in self.depletions:
                self.depletions.remove(edge)

    def _extend_case_i(self, e: int, cur_queue: float):
        capacity, travel_time = self._network.capacity[e], self._network.travel_time[e]
        arrival = self.phi + cur_queue / capacity + travel_time

        for i in range(len(self._network.commodities)):
            self.inflow[e][i].extend(self.phi, 0)
            self.outflow[e][i].extend(arrival, 0)

        self.outflow_changes.set((e, arrival), arrival)

        queue_slope = 0. if cur_queue == 0. else -capacity
        self.queues[e].extend_with_slope(self.phi, queue_slope)
        if cur_queue > 0:
            depl_time = self.phi + cur_queue / capacity
            assert self.queues[e](depl_time) < 1000 * eps
            self.depletions.set(e, depl_time)
        elif e in self.depletions:
            self.depletions.remove(e)

    def _extend_case_ii(self, e: int, new_inflow: List[float], cur_queue: float, acc_in: float):
        capacity, travel_time = self._network.capacity[e], self._network.travel_time[e]
        arrival = self.phi + cur_queue / capacity + travel_time

        for i in range(len(self._network.commodities)):
            self.inflow[e][i].extend(self.phi, new_inflow[i])
            new_out = min(capacity, acc_in) * new_inflow[i] / acc_in
            self.outflow[e][i].extend(arrival, new_out)

        self.outflow_changes.set((e, arrival), arrival)

        queue_slope = max(acc_in - capacity, 0.)
        self.queues[e].extend_with_slope(self.phi, queue_slope)
        if e in self.depletions:
            self.depletions.remove(e)

    def _extend_case_iii(self, e: int, new_inflow: List[float], cur_queue: float, acc_in: float):
        capacity, travel_time = self._network.capacity[e], self._network.travel_time[e]
        arrival = self.phi + cur_queue / capacity + travel_time

        for i in range(len(self._network.commodities)):
            self.inflow[e][i].extend(self.phi, new_inflow[i])
            new_out = capacity * new_inflow[i] / acc_in
            self.outflow[e][i].extend(arrival, new_out)

        self.outflow_changes.set((e, arrival), arrival)

        queue_slope = acc_in - capacity
        self.queues[e].extend_with_slope(self.phi, queue_slope)

        depl_time = self.phi - cur_queue / queue_slope
        planned_change = depl_time + travel_time
        assert self.queues[e](depl_time) < 1000 * eps
        self.depletions.set(e, depl_time, (planned_change, new_inflow))

    def _process_depletions(self):
        while self.depletions.min_depletion() <= self.phi:
            (e, depl_time, change_event) = self.depletions.pop_by_depletion()
            self.queues[e].extend_with_slope(depl_time, 0.)
            assert abs(self.queues[e].values[-1]) < 1000 * eps
            self.queues[e].values[-1] = 0.
            if change_event is not None:
                self.outflow_changes.set((e, change_event[0]), change_event[0])
                for i in range(len(self._network.commodities)):
                    self.outflow[e][i].extend(change_event[0], change_event[1][i])

    def extend(self, new_inflow: Dict[int, List[float]], max_extension_time: float) -> Set[int]:
        """
        Extends the flow with constant inflows new_inflow until some edge outflow changes.
        Edge inflows not in new_inflow are extended with their previous values.
        The user can also specify a maximum extension length using max_extension_length.
        :returns set of edges where the outflow has changed
        """
        capacity = self._network.capacity

        for e in new_inflow.keys():
            if all(new_inflow[e][i] == self.inflow[e][i].values[-1] for i in range(len(self._network.commodities))):
                continue
            acc_in, cur_queue = sum(new_inflow[e]), max(self.queues[e](self.phi), 0.)

            if acc_in == 0.:
                self._extend_case_i(e, cur_queue)
            elif cur_queue == 0. or acc_in >= capacity[e]:
                self._extend_case_ii(e, new_inflow[e], cur_queue, acc_in)
            else:
                self._extend_case_iii(e, new_inflow[e], cur_queue, acc_in)

        self.phi = min(
            self.depletions.min_change_time(),
            self.outflow_changes.min_key(),
            max_extension_time
        )

        self._process_depletions()

        changed_edges: Set[int] = set()
        while self.outflow_changes.min_key() <= self.phi:
            changed_edges.add(self.outflow_changes.pop()[0])
        return changed_edges

    def avg_travel_time(self, i: int, horizon: float) -> float:
        commodity = self._network.commodities[i]
        net_outflow: RightConstant = sum(self.outflow[e.id][i] for e in commodity.sink.incoming_edges)
        accum_net_outflow = net_outflow.integral()
        accum_net_inflow = commodity.net_inflow.integral()

        avg_travel_time = \
            (accum_net_inflow.integrate(0., horizon) - accum_net_outflow.integrate(0., horizon)) / accum_net_inflow(
                horizon)
        return avg_travel_time
