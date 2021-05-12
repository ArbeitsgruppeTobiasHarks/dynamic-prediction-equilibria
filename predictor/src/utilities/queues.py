import heapq
from dataclasses import dataclass
from typing import Optional, List, TypeVar, Generic

from core.machine_precision import eps

T = TypeVar('T')


@dataclass
class PriorityItem(Generic[T]):
    key: float
    item: T


class PriorityQueue(Generic[T]):
    def __init__(self, initial: Optional[List[PriorityItem[T]]] = None):
        self.index = 0  # to avoid clashes with key
        if initial:
            self._data = [(item.key, i, item.item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item: PriorityItem[T]):
        heapq.heappush(self._data, (item.key, self.index, item.item))
        self.index += 1

    def pop(self) -> T:
        return heapq.heappop(self._data)[2]

    def next(self) -> T:
        return self._data[0][2]

    def decrease_key(self, item: T, key: float):
        # Todo: The performance of this operation can be improved.
        for i in range(len(self._data)):
            if self._data[i][2] == item:
                assert key <= self._data[i][0] + eps
                self._data[i] = (min(self._data[i][0], key), self._data[i][1], item)
        heapq.heapify(self._data)

    def min_time(self) -> float:
        return float('inf') if len(self._data) == 0 else self._data[0][0]

    def sorted(self) -> List[T]:
        return [ev_tuple[2] for ev_tuple in heapq.nsmallest(len(self._data), self._data)]

    def __len__(self) -> int:
        return self._data.__len__()
