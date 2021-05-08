from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from utilities.arrays import elem_rank


@dataclass
class LinearlyInterpolatedFunction:
    times: List[float]
    values: List[float]
    domain: Tuple[float, float] = (float('-inf'), float('inf'))

    def __call__(self, at: float) -> float:
        return self.eval(at)

    def eval(self, at: float) -> float:
        assert self.domain[0] <= at <= self.domain[1], f"Function not defined at {at}."
        rnk = elem_rank(self.times, at)
        if rnk == -1:
            first_grad = self.gradient(0)
            if at == float('-inf') and first_grad == 0:
                return self.values[0]
            else:
                return self.values[0] + (at - self.times[0]) * self.gradient(0)
        elif rnk == len(self.times) - 1:
            last_grad = self.gradient(-2)
            if at == float('inf') and last_grad == 0:
                return self.values[-1]
            else:
                return self.values[-1] + (at - self.times[-1]) * self.gradient(-2)
        return self.values[rnk] + (at - self.times[rnk]) * self.gradient(rnk)

    def gradient(self, i: int) -> float:
        """
            Returns the gradient between times[i] and times[i+1]
        """
        return (self.values[i + 1] - self.values[i]) / (self.times[i + 1] - self.times[i])

    def plus(self, other: LinearlyInterpolatedFunction) -> LinearlyInterpolatedFunction:
        """
            Calculate the sum of two functions.
            Can still be optimized: There might be unnecessary time-steps at the boundaries.
        """
        new_domain = (
            max(self.domain[0], other.domain[0]),
            min(self.domain[1], other.domain[1])
        )
        assert new_domain[0] < new_domain[1], "Intersection of function domains is empty."

        my_ind = max(0, elem_rank(self.times, new_domain[0]))
        other_ind = max(0, elem_rank(other.times, new_domain[0]))
        times = [min(self.times[my_ind], other.times[other_ind])]
        last_time = min(new_domain[1], max(self.times[-1], other.times[-1]))
        while times[-1] < last_time:
            if my_ind < len(self.times) and self.times[my_ind] <= new_domain[1]:
                if self.times[my_ind] > times[-1]:
                    times.append(self.times[my_ind])
                my_ind += 1
            elif other_ind < len(other.times) and other.times[other_ind] <= new_domain[1]:
                if other.times[other_ind] > times[-1]:
                    times.append(other.times[other_ind])
                other_ind += 1
        values: List[float] = [self(phi) + other(phi) for phi in times]
        return LinearlyInterpolatedFunction(times, values, new_domain)

    def inverse(self, x: float, i: int) -> float:
        assert 0 <= i < len(self.times)
        assert self.values[i] < self.values[i + 1] or self.values[i] > self.values[i + 1], \
            "Can only determine inverse on strictly monotone interval"
        assert self.values[i] <= x <= self.values[i + 1] or self.values[i] >= x >= self.values[i + 1], \
            "x must be between values[i] and values[i+1]"
        lmbda = (x - self.values[i + 1]) / (self.values[i] - self.values[i + 1])
        return lmbda * self.times[i] + (1 - lmbda) * self.times[i + 1]

    def compose(self, f: LinearlyInterpolatedFunction) -> LinearlyInterpolatedFunction:
        g = self
        # We calculate phi -> g( f( phi ) )
        assert self.is_monotone() and f.is_monotone(), "Composition only implemented for monotone incr. functions"
        assert f.domain[0] > float('-inf'), "Composition only implemented for left-finite intervals"
        assert g.domain[0] <= f.image()[0] and g.domain[1] >= f.image()[1], \
            "The domains do not match for composition!"

        assert f.domain[0] == f.times[0] and f.domain[1] == f.times[-1], "Assert domain are numbers in times"

        times = []

        f_ind = 0  # Start of analyzed interval
        g_ind = max(0, elem_rank(g.times, f(f.domain[0])))  # Start of interval
        assert g.times[g_ind] <= f.values[f_ind] <= g.times[g_ind + 1]

        while f_ind < len(f.times) - 1:
            f_after = f.values[f_ind + 1]

            while g_ind < len(g.times) and g.times[g_ind] <= f_after:
                times.append(f.inverse(g.times[g_ind], f_ind))
                g_ind += 1
            if f.times[f_ind + 1] > times[-1]:
                times.append(f.times[f_ind + 1])
            f_ind += 1
        values: List[float] = [g(f(phi)) for phi in times]
        return LinearlyInterpolatedFunction(times, values, f.domain)

    def is_monotone(self):
        return all(self.values[i] < self.values[i + 1] for i in range(len(self.values) - 1))

    def image(self) -> Tuple[float, float]:
        assert self.is_monotone(), "Only implented for monotone functions"
        return self(self.domain[0]), self(self.domain[1])
