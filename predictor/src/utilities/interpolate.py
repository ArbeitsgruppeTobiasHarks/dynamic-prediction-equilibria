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
            first_grad = self.gradient(rnk)
            if at == float('-inf') and first_grad == 0:
                return self.values[0]
            else:
                return self.values[0] + (at - self.times[0]) * first_grad
        elif rnk == len(self.times) - 1:
            last_grad = self.gradient(rnk)
            if at == float('inf') and last_grad == 0:
                return self.values[-1]
            else:
                return self.values[-1] + (at - self.times[-1]) * last_grad
        return self.values[rnk] + (at - self.times[rnk]) * self.gradient(rnk)

    def gradient(self, i: int) -> float:
        """
            Returns the gradient between times[i] (or domain[0] if i == -1)
            and times[i+1] (or domain[1] if i == len(times) - 1)
        """
        assert -1 <= i < len(self.times)
        if i == -1:
            i = 0
        elif i == len(self.times) - 1:
            i = len(self.times) - 2
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
        assert -1 <= i < len(self.times)
        if i == -1:
            assert (self.gradient(i) > 0 and x <= self.values[0]) \
                   or (self.gradient(i) < 0 and x >= self.values[0])
            return self.times[0] + (x - self.values[0]) / self.gradient(i)
        elif i == len(self.times) - 1:
            assert (self.gradient(i) > 0 and x >= self.values[0]) \
                   or (self.gradient(i) < 0 and x <= self.values[0])
            return self.times[-1] + (x - self.values[-1]) / self.gradient(i)
        assert self.values[i] < self.values[i + 1] or self.values[i] > self.values[i + 1], \
            "Can only determine inverse on strictly monotone interval"
        assert self.values[i] <= x <= self.values[i + 1] or self.values[i] >= x >= self.values[i + 1], \
            "x must be between values[i] and values[i+1]"
        lmbda = (x - self.values[i + 1]) / (self.values[i] - self.values[i + 1])
        return lmbda * self.times[i] + (1 - lmbda) * self.times[i + 1]

    def compose(self, f: LinearlyInterpolatedFunction) -> LinearlyInterpolatedFunction:
        g = self
        # We calculate g ⚬ f
        assert f.is_monotone(), "Composition g ⚬ f only implemented for monotone incr. function f"
        assert g.domain[0] <= f.image()[0] and g.domain[1] >= f.image()[1], \
            "The domains do not match for composition!"

        times = []

        f_ind = -1  # Start of analyzed interval
        g_ind = max(0, elem_rank(g.times, f(f.domain[0])))  # Start of interval
        assert g_ind == len(g.times) - 1 or f.domain[0] <= g.times[g_ind + 1]

        while f_ind < len(f.times):
            f_after = f.values[f_ind + 1] if f_ind < len(f.times) - 1 else f(f.domain[1])

            while g_ind < len(g.times) and g.times[g_ind] <= f_after:
                next_time = max(f(f.domain[0]), g.times[g_ind])
                inverse = f.inverse(next_time, f_ind)
                if len(times) == 0 or inverse > times[-1]:
                    times.append(inverse)
                g_ind += 1
            if f_ind + 1 < len(f.times):
                if len(times) == 0 or f.times[f_ind + 1] > times[-1]:
                    times.append(f.times[f_ind + 1])
            f_ind += 1

        values: List[float] = [g(f(phi)) for phi in times]
        return LinearlyInterpolatedFunction(times, values, f.domain)

    def minimum(self, otherf: LinearlyInterpolatedFunction) -> LinearlyInterpolatedFunction:
        # Calculate the pointwise minimum of self and otherf.
        # TODO: This procedure is by not perfect yet.
        new_domain = (max(self.domain[0], otherf.domain[0]), min(self.domain[1], otherf.domain[1]))
        assert new_domain[0] < new_domain[1], "Intersection of function domains is empty."
        assert new_domain[0] in self.times or new_domain[0] in otherf.times, \
            "Overtaking before the first point is not handled yet."

        f = [self, otherf]
        curr_min = 0 if f[0](new_domain[0]) < f[1](new_domain[0]) else 1
        other = 1 - curr_min
        ind = [0, 0]
        times = []
        while ind[0] < len(f[0].times) or ind[1] < len(f[1].times):
            if ind[other] >= len(f[other].times):
                fct = curr_min
            elif ind[curr_min] >= len(f[curr_min].times):
                fct = other
            elif f[other].times[ind[other]] <= f[curr_min].times[ind[curr_min]]:
                fct = other
            else:
                fct = curr_min
            next_time = f[fct].times[ind[fct]]
            if next_time > new_domain[1]:
                break
            if fct != curr_min:
                curr_min_val = f[curr_min](next_time)
                curr_other_val = f[fct].values[ind[fct]]
            else:
                curr_other_val = f[other](next_time)
                curr_min_val = f[fct].values[ind[fct]]
            if curr_other_val < curr_min_val:
                # The minimum function has changed!
                # Find the intersecting time with x=next_time:
                # t = x + (g(x) - f(x)) / (grad_f - grad_g)
                grad_min = f[curr_min].gradient(ind[curr_min] - 1)
                grad_other = f[other].gradient(ind[other] - 1)
                difference = grad_min - grad_other
                t = next_time + (curr_other_val - curr_min_val) / difference
                if len(times) == 0 or t > times[-1]:
                    times.append(t)
                curr_min = fct
                other = 1 - fct
            if fct == curr_min and (len(times) == 0 or next_time > times[-1]):
                times.append(next_time)
            ind[fct] += 1

        if times[-1] < new_domain[1]:
            grad_min = f[curr_min].gradient(ind[curr_min] - 1)
            grad_other = f[other].gradient(ind[other] - 1)
            if grad_min > grad_other:
                curr_min_val = f[curr_min](times[-1])
                curr_other_val = f[other](times[-1])
                difference = grad_min - grad_other
                t = times[-1] + (curr_other_val - curr_min_val) / difference
                if t <= new_domain[1]:
                    # Min function has changed once again. We need another two points to adjust the gradient.
                    times.append(t)
                    if t < new_domain[1]:
                        times.append(t + 1 if new_domain[1] == float('inf') else new_domain[1])

        values = [min(self(t), otherf(t)) for t in times]
        return LinearlyInterpolatedFunction(times, values, new_domain)

    def is_monotone(self):
        return all(self.values[i] < self.values[i + 1] for i in range(len(self.values) - 1))

    def image(self) -> Tuple[float, float]:
        assert self.is_monotone(), "Only implemented for monotone functions"
        return self(self.domain[0]), self(self.domain[1])


identity = LinearlyInterpolatedFunction([0., 1.], [0., 1.])
