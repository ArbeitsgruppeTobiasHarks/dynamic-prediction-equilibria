import heapq
from typing import Optional, List, TypeVar, Generic


class Event:
    def __init__(self, time: float):
        self.time = time


T = TypeVar('T', bound=Event)


class PriorityQueue(Generic[T]):
    def __init__(self, initial: Optional[List[T]] = None):
        self.index = 0  # to avoid clashes with key
        if initial:
            self._data = [(item.time, i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (item.time, self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self._data)[2]

    def min_time(self):
        return float('inf') if len(self._data) == 0 else self._data[0][0]

    def sorted(self):
        return [ev_tuple[2] for ev_tuple in heapq.nsmallest(len(self._data), self._data)]

    def __len__(self):
        return self._data.__len__()

