from __future__ import annotations

from typing import Dict, Optional
from utilities.arrays import merge_sorted
from utilities.piecewise_linear import PiecewiseLinear
from core.machine_precision import eps

from utilities.right_constant import RightConstant


class FlowRatesCollectionItem:
    time: float
    values: Dict[int, float]
    next_item: Optional[FlowRatesCollectionItem]

    def __init__(self, time: float, values: Dict[int, float], next_item=None):
        assert all(value > 0 for value in values.values())
        self.time = time
        self.values = values
        self.next_item = next_item


class FlowRatesCollection:
    _functions_dict: Dict[int, RightConstant]
    _queue_head: Optional[FlowRatesCollectionItem]
    _queue_tail: Optional[FlowRatesCollectionItem]
    accumulative: PiecewiseLinear

    def __init__(self, functions_dict: Optional[Dict[int, RightConstant]] = None):
        self._functions_dict = {} if functions_dict is None else functions_dict
        self._queue_head = FlowRatesCollectionItem(0.0, {})
        self._queue_tail = self._queue_head
        self.accumulative = RightConstant.sum(
            list(self._functions_dict.values())
        ).integral()
        times = []
        for fun in self._functions_dict.values():
            times = merge_sorted(times, fun.times)
        for time in times:
            item = FlowRatesCollectionItem(
                time,
                {
                    i: fun(time)
                    for i, fun in self._functions_dict.items()
                    if fun(time) > 0.0
                },
            )
            self._queue_tail.next_item = item
            self._queue_tail = item

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Don't pickle _network b.c. of recursive structure
        del state["_queue_tail"]
        del state["_queue_head"]
        unrolled_queue = []

        curr_item = self._queue_head
        while curr_item is not None:
            unrolled_queue.append({"time": curr_item.time, "values": curr_item.values})
            curr_item = curr_item.next_item
        state["_queue"] = unrolled_queue
        return state

    def __setstate__(self, state):
        queue = state["_queue"]
        del state["_queue"]
        self.__dict__.update(state)
        if len(queue) == 0:
            self._queue_head = None
            self._queue_tail = None
        else:
            self._queue_head = FlowRatesCollectionItem(
                queue[0]["time"], queue[0]["values"]
            )
            self._queue_tail = self._queue_head
            for i in range(1, len(queue)):
                next_item = FlowRatesCollectionItem(
                    queue[i]["time"], queue[i]["values"]
                )
                self._queue_tail.next_item = next_item
                self._queue_tail = next_item

    def extend(self, time: float, values: Dict[int, float], values_sum: float):
        item = FlowRatesCollectionItem(time, values)
        if self._queue_tail is None:
            self._queue_head = item
            self._queue_tail = item
            for i, value in values.items():
                self._functions_dict[i] = FlowRatesCollection._new_flow_fn()
                self._functions_dict[i].extend(time, value)
        else:
            assert self._queue_tail.time <= time + eps
            for i, value in values.items():
                if i not in self._functions_dict:
                    self._functions_dict[i] = FlowRatesCollection._new_flow_fn()
                self._functions_dict[i].extend(time, value)
            for i in self._queue_tail.values:
                if i not in values:
                    self._functions_dict[i].extend(time, 0.0)
            self._queue_tail.next_item = item
            self._queue_tail = item
        self.accumulative.extend_with_slope(time, values_sum)

    def get_values_at_time(self, time: float) -> Dict[int, float]:
        if self._queue_head is None:
            return {}
        elif self._queue_head.time > time:
            raise ValueError("The desired time is not available anymore.")
        else:
            while (
                self._queue_head.next_item is not None
                and self._queue_head.next_item.time <= time
            ):
                self._queue_head = self._queue_head.next_item
            return self._queue_head.values

    @staticmethod
    def _new_flow_fn():
        return RightConstant([0.0], [0.0], (0.0, float("inf")))
