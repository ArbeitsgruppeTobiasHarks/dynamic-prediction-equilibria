import heapq
from typing import Optional, List, TypeVar, Generic, Tuple, Dict, Hashable

T = TypeVar('T', bound=Hashable)


class PriorityQueue(Generic[T]):
    """
    This is a min-heap queue with decrease-key operation.
    The implementation of many functions is taken from pythons heapq library.
    """

    _data: List[Tuple[float, int, T]]
    _index_dict: Dict[T, int]
    _index: int

    def __init__(self, initial: Optional[List[Tuple[T, float]]] = None):
        if initial:
            self._data = [(key, i, item) for i, (item, key) in enumerate(initial)]
            self._index = len(self._data)
            heapq.heapify(self._data)
            self._index_dict = {item: i for i, (_, _, item) in enumerate(self._data)}
        else:
            self._data = []
            self._index = 0
            self._index_dict = {}

    def push(self, item: T, key: float):
        new_entry = (key, self._index, item)
        self._data.append(new_entry)
        self._index_dict[item] = len(self._data) - 1
        self._siftdown(0, len(self._data) - 1)
        self._index += 1

    def _siftdown(self, startpos, pos):
        newitem = self._data[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self._data[parentpos]
            if newitem < parent:
                self._data[pos] = parent
                self._index_dict[parent[2]] = pos
                pos = parentpos
                continue
            break
        self._data[pos] = newitem
        self._index_dict[newitem[2]] = pos

    def pop(self) -> T:
        lastelt = self._data.pop()  # raises appropriate IndexError if heap is empty
        self._index_dict.pop(lastelt[2])

        if self._data:
            returnitem = self._data[0]
            self._index_dict.pop(returnitem[2])

            self._data[0] = lastelt
            self._index_dict[lastelt[2]] = 0
            self._siftup(0)
            return returnitem[2]
        return lastelt[2]

    def _siftup(self, pos):
        endpos = len(self._data)
        startpos = pos
        newitem = self._data[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self._data[childpos] < self._data[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self._data[pos] = self._data[childpos]
            self._index_dict[self._data[pos][2]] = pos

            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self._data[pos] = newitem
        self._index_dict[newitem[2]] = pos
        self._siftdown(startpos, pos)

    def next(self) -> T:
        return self._data[0][2]

    def key_of(self, item: T, default: Optional[float] = None) -> float:
        if item not in self._index_dict:
            return default
        return self._data[self._index_dict[item]][0]

    def decrease_key(self, item: T, key: float):
        assert item in self._index_dict.keys()
        pos = self._index_dict[item]
        self._data[pos] = (key, self._data[pos][1], item)
        self._siftdown(0, pos)

    def min_key(self) -> float:
        return float('inf') if len(self._data) == 0 else self._data[0][0]

    def sorted(self) -> List[T]:
        return [ev_tuple[2] for ev_tuple in heapq.nsmallest(len(self._data), self._data)]

    def __len__(self) -> int:
        return self._data.__len__()
