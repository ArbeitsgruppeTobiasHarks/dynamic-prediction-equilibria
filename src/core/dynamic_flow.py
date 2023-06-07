from __future__ import annotations
from functools import lru_cache

from typing import List, Dict, Set, Tuple, Optional
from core.flow_rates_collection import FlowRatesCollection

from core.machine_precision import eps
from core.network import Network
from utilities.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue
from utilities.right_constant import RightConstant

ChangeEventValue = Tuple[Dict[int, float], float]
ChangeEvent = Optional[Tuple[float, ChangeEventValue]]


class DepletionQueue:
    depletions: PriorityQueue[int]
    change_times: PriorityQueue[int]
    new_outflow: Dict[int, ChangeEventValue]  # time, comm: outflow, sum over outflow

    def __init__(self):
        self.depletions = PriorityQueue()
        self.change_times = PriorityQueue()
        self.new_outflow = {}

    def set(
        self, edge: int, depletion_time: float, change_event: ChangeEvent = None
    ) -> None:
        assert depletion_time > float("-inf")
        self.depletions.set(edge, depletion_time)

        if change_event is not None:
            (change_time, change_value) = change_event
            self.new_outflow[edge] = change_value
            self.change_times.set(edge, change_time)
        elif edge in self.change_times:
            self.change_times.remove(edge)
            self.new_outflow.pop(edge)

    def __contains__(self, edge) -> bool:
        return edge in self.depletions

    def remove(self, edge: int) -> None:
        self.depletions.remove(edge)
        if edge in self.change_times:
            self.change_times.remove(edge)
            self.new_outflow.pop(edge)

    def min_change_time(self) -> float:
        return self.change_times.min_key()

    def min_depletion(self) -> float:
        return self.depletions.min_key()

    def pop_by_depletion(self) -> Tuple[int, float, ChangeEvent]:
        depl_time, e = self.depletions.min_key(), self.depletions.pop()
        change_event = None
        if e in self.change_times:
            change_time = self.change_times.key_of(e)
            self.change_times.remove(e)
            new_outflow, new_outflow_sum = self.new_outflow.pop(e)
            change_event = (change_time, (new_outflow, new_outflow_sum))
        return e, depl_time, change_event


class DynamicFlow:
    """
    This is a representation of a flow with right-constant edge inflow rates on intervals.
    """

    phi: float
    # inflow[e][i] is the function fᵢₑ⁺
    inflow: List[FlowRatesCollection]
    # outflow[e][i] is the function fᵢₑ⁻
    outflow: List[FlowRatesCollection]
    queues: List[PiecewiseLinear]  # queues[e] is the queue length at e
    # A priority queue with times when some edge outflow changes
    outflow_changes: PriorityQueue[Tuple[int, float]]
    depletions: DepletionQueue  # A priority queue with events at which queues deplete
    _network: Network

    def __init__(self, network: Network):
        self._network = network
        self.phi = 0.0
        self.inflow = [FlowRatesCollection() for _ in network.graph.edges]
        self.queues = [
            PiecewiseLinear([self.phi], [0.0], 0.0, 0.0) for _ in network.graph.edges
        ]
        self.outflow = [FlowRatesCollection() for _ in network.graph.edges]
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

    def _extend_case_i(self, e: int, cur_queue: float):
        capacity, travel_time = self._network.capacity[e], self._network.travel_time[e]
        arrival = self.phi + cur_queue / capacity + travel_time

        self.outflow[e].extend(arrival, {}, 0.0)

        self.outflow_changes.set((e, arrival), arrival)

        queue_slope = 0.0 if cur_queue == 0.0 else -capacity
        self.queues[e].extend_with_slope(self.phi, queue_slope)
        if cur_queue > 0:
            depl_time = self.phi + cur_queue / capacity
            assert self.queues[e](depl_time) < 1000 * eps
            self.depletions.set(e, depl_time)
        elif e in self.depletions:
            self.depletions.remove(e)

    def _extend_case_ii(
        self, e: int, new_inflow: Dict[int, float], cur_queue: float, acc_in: float
    ):
        capacity, travel_time = self._network.capacity[e], self._network.travel_time[e]
        arrival = self.phi + cur_queue / capacity + travel_time

        acc_out = min(capacity, acc_in)
        factor = acc_out / acc_in

        new_outflow = {i: factor * value for i, value in new_inflow.items()}
        self.outflow[e].extend(arrival, new_outflow, acc_out)

        self.outflow_changes.set((e, arrival), arrival)

        queue_slope = max(acc_in - capacity, 0.0)
        self.queues[e].extend_with_slope(self.phi, queue_slope)
        if e in self.depletions:
            self.depletions.remove(e)

    def _extend_case_iii(
        self, e: int, new_inflow: Dict[int, float], cur_queue: float, acc_in: float
    ):
        capacity, travel_time = self._network.capacity[e], self._network.travel_time[e]
        arrival = self.phi + cur_queue / capacity + travel_time

        factor = capacity / acc_in

        new_outflow = {i: factor * value for i, value in new_inflow.items()}
        self.outflow[e].extend(arrival, new_outflow, capacity)

        self.outflow_changes.set((e, arrival), arrival)

        queue_slope = acc_in - capacity
        self.queues[e].extend_with_slope(self.phi, queue_slope)

        depl_time = self.phi - cur_queue / queue_slope
        planned_change_time = depl_time + travel_time
        planned_change_value = (new_inflow, acc_in)
        assert self.queues[e](depl_time) < 1000 * eps

        self.depletions.set(e, depl_time, (planned_change_time, planned_change_value))

    def _process_depletions(self):
        while self.depletions.min_depletion() <= self.phi:
            (e, depl_time, change_event) = self.depletions.pop_by_depletion()
            self.queues[e].extend_with_slope(depl_time, 0.0)
            assert abs(self.queues[e].values[-1]) < 1000 * eps
            self.queues[e].values[-1] = 0.0
            if change_event is not None:
                (change_time, (new_outflow, new_outflow_sum)) = change_event
                self.outflow_changes.set((e, change_time), change_time)
                self.outflow[e].extend(change_time, new_outflow, new_outflow_sum)

    def extend(
        self, new_inflow: Dict[int, Dict[int, float]], max_extension_time: float
    ) -> Set[int]:
        """
        Extends the flow with constant inflows new_inflow until some edge outflow changes.
        Edge inflows not in new_inflow are extended with their previous values.
        The user can also specify a maximum extension length using max_extension_length.
        :returns set of edges where the outflow has changed at the new time self.phi
        """
        self.get_edge_loads.cache_clear()
        capacity = self._network.capacity

        for e in new_inflow.keys():
            if self.inflow[e].get_values_at_time(self.phi) == new_inflow[e]:
                continue
            acc_in = sum(new_inflow[e].values())
            cur_queue = max(self.queues[e].eval_from_end(self.phi), 0.0)

            self.inflow[e].extend(self.phi, new_inflow[e], acc_in)
            if acc_in == 0.0:
                self._extend_case_i(e, cur_queue)
            elif cur_queue == 0.0 or acc_in >= capacity[e] - eps:
                self._extend_case_ii(e, new_inflow[e], cur_queue, acc_in)
            else:
                self._extend_case_iii(e, new_inflow[e], cur_queue, acc_in)

        self.phi = min(
            self.depletions.min_change_time(),
            self.outflow_changes.min_key(),
            max_extension_time,
        )

        self._process_depletions()

        changed_edges: Set[int] = set()
        while self.outflow_changes.min_key() <= self.phi:
            changed_edges.add(self.outflow_changes.pop()[0])
        return changed_edges

    def avg_travel_time(self, i: int, horizon: float) -> float:
        commodity = self._network.commodities[i]
        net_outflow: RightConstant = sum(
            (
                self.outflow[e.id]._functions_dict[i]
                for e in commodity.sink.incoming_edges
                if i in self.outflow[e.id]._functions_dict
            ),
            start=RightConstant([0.0], [0.0], (0, float("inf"))),
        )
        accum_net_outflow = net_outflow.integral()
        net_inflow: RightConstant = sum(
            (inflow for inflow in commodity.sources.values()),
            start=RightConstant([0.0], [0.0], (0, float("inf"))),
        )
        accum_net_inflow = net_inflow.integral()

        avg_travel_time = (
            accum_net_inflow.integrate(0.0, horizon)
            - accum_net_outflow.integrate(0.0, horizon)
        ) / accum_net_inflow(horizon)
        return avg_travel_time

    @lru_cache()
    def get_edge_loads(self) -> List[PiecewiseLinear]:
        edge_loads: List[PiecewiseLinear] = [
            self.inflow[e].accumulative - self.outflow[e].accumulative
            for e in range(len(self.inflow))
        ]
        assert all(edge_load.domain[0] == 0.0 for edge_load in edge_loads)
        assert all(abs(edge_load(0.0)) < 1e-10 for edge_load in edge_loads)
        for edge_load in edge_loads:
            if edge_load.values[0] != 0.0:
                edge_load.times.insert(0, 0.0)
                edge_load.values.insert(0, 0.0)
            edge_load.first_slope = 0.0
            edge_load.domain = (float("-inf"), edge_load.domain[1])

        return edge_loads
