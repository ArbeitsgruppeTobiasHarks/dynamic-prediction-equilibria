from __future__ import annotations

from typing import List, Tuple

from core.machine_precision import eps
from utilities.arrays import elem_lrank, merge_sorted
from utilities.interpolate import LinearlyInterpolatedFunction


class RightConstantFunction:
    """
    This class defines right-continuous functions with finitely many break points (xᵢ, yᵢ).
    The breakpoints are encoded in the two lists times = [x₀, ..., xₙ] and values = [y₀,..., yₙ]
    where we assume that times is strictly increasing.
    This encodes the function f(x) = yᵢ with i maximal s.t. xᵢ <= x (or i = 0 if x₀ > x).
    """

    times: List[float]
    values: List[float]
    domain: Tuple[float, float] = (float('-inf'), float('inf'))

    def __init__(self, times: List[float], values: List[float],
                 domain: Tuple[float, float] = (float('-inf'), float('inf'))):
        self.times = times
        self.values = values
        self.domain = domain
        assert len(self.values) == len(self.times)
        assert all(float('-inf') < self.values[i] < float('inf') for i in range(len(self.times)))
        assert all(self.domain[0] <= self.times[i] <= self.domain[1] for i in range(len(self.times)))

    def __call__(self, at: float) -> float:
        return self.eval(at)

    def eval(self, at: float) -> float:
        assert self.domain[0] <= at <= self.domain[1], f"Function not defined at {at}."
        rnk = elem_lrank(self.times, at)
        return self._eval_with_lrank(at, rnk)

    def _eval_with_lrank(self, at: float, rnk: int):
        assert self.domain[0] <= at <= self.domain[1], f"Function not defined at {at}."
        assert -1 <= rnk <= len(self.times)
        assert rnk == elem_lrank(self.times, at)

        if rnk == -1:
            return self.values[0]
        else:
            return self.values[rnk]

    def extend(self, start_time: float, value: float):
        assert start_time >= self.times[-1] - eps
        if start_time <= self.times[-1] + eps:
            #  Simply replace the last value
            self.values[-1] = value
        elif self.values[-1] != value:
            self.times.append(start_time)
            self.values.append(value)

    def equals(self, other):
        if not isinstance(other, RightConstantFunction):
            return False
        return self.values == other.values and self.times == other.times and self.domain == other.domain

    def __radd__(self, other):
        if other == 0:
            return self
        if not isinstance(other, RightConstantFunction):
            raise TypeError("Can only add a RightConstantFunction.")
        assert self.domain == other.domain

        new_times = merge_sorted(self.times, other.times)
        new_values = [self(t) + other(t) for t in new_times]
        return RightConstantFunction(new_times, new_values, self.domain)

    def __add__(self, other):
        return self.__radd__(other)

    def integral(self) -> LinearlyInterpolatedFunction:
        assert self.times[0] == self.domain[0] and self.domain[1] >= self.times[-1] + 1
        times = self.times + [self.times[-1] + 1]
        values = [0.] * len(times)
        for i in range(len(times) - 1):
            values[i + 1] = values[i] + self.values[i] * (times[i + 1] - times[i])
        return LinearlyInterpolatedFunction(times, values, self.domain)
