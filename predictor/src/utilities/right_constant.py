from __future__ import annotations

from typing import List, Tuple
import json_fix

from core.machine_precision import eps
from utilities.arrays import elem_lrank, merge_sorted
from utilities.piecewise_linear import PiecewiseLinear

json_fix.fix_it()


class RightConstant:
    """
    This class defines right-continuous functions with finitely many break points (xᵢ, yᵢ).
    The breakpoints are encoded in the two lists times = [x₀, ..., xₙ] and values = [y₀,..., yₙ]
    where we assume that times is strictly increasing.
    This encodes the function f(x) = yᵢ with i maximal s.t. xᵢ <= x (or i = 0 if x₀ > x).
    """

    times: List[float]
    values: List[float]
    domain: Tuple[float, float] = (float('-inf'), float('inf'))

    def __json__(self):
        return {
            "times": self.times,
            "values": self.values,
            "domain": [
                '-Infinity' if self.domain[0] == float('-inf') else self.domain[0],
                'Infinity' if self.domain[1] == float('inf') else self.domain[1]
            ]
        }

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
        if abs(self.values[-1] - value) <= eps:
            return
        if abs(start_time - self.times[-1]) <= eps:
            #  Simply replace the last value
            self.values[-1] = value
        else:
            self.times.append(start_time)
            self.values.append(value)

    def equals(self, other):
        if not isinstance(other, RightConstant):
            return False
        return self.values == other.values and self.times == other.times and self.domain == other.domain

    def __radd__(self, other):
        if other == 0:
            return self
        if not isinstance(other, RightConstant):
            raise TypeError("Can only add a RightConstantFunction.")
        assert self.domain == other.domain

        new_times = merge_sorted(self.times, other.times)
        new_values = [self(t) + other(t) for t in new_times]
        return RightConstant(new_times, new_values, self.domain)
    
    def __add__(self, other):
        return self.__radd__(other)
    
    def __neg__(self):
        return RightConstant(self.times, [-v for v in self.values], self.domain)

    def __sub__(self, other):
        if not isinstance(other, RightConstant):
            raise TypeError("Can only subtract a RightConstantFunction.")
        return self + (-other)
    
    def simplify(self) -> RightConstant:
        """
        This removes unnecessary timesteps
        """
        new_times = [self.times[0]]
        new_values = [self.values[0]]
        for i in range(0, len(self.times) - 1):
            # Add i+1, if it's necessary.
            if abs(self.values[i] - self.values[i + 1]) >= 1000*eps:
                new_times.append(self.times[i + 1])
                new_values.append(self.values[i + 1])
        return RightConstant(new_times, new_values, self.domain)

    def integral(self) -> PiecewiseLinear:
        """
        Returns the integral starting from self.times[0] to x of self.
        """
        assert self.times[0] == self.domain[0] and self.domain[1] >= self.times[-1] + 1
        times = self.times
        values = [0.] * len(times)
        for i in range(len(times) - 1):
            values[i + 1] = values[i] + self.values[i] * (times[i + 1] - times[i])
        return PiecewiseLinear(times, values, self.values[0], self.values[-1], self.domain)

    