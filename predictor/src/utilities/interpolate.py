from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Optional

from core.machine_precision import eps
from utilities.arrays import elem_rank, elem_lrank, merge_sorted


class LinearlyInterpolatedFunction:
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
        rnk = elem_rank(self.times, at)
        return self._eval_with_rank(at, rnk)

    def _eval_with_rank(self, at: float, rnk: int):
        assert self.domain[0] <= at <= self.domain[1], f"Function not defined at {at}."
        assert -1 <= rnk <= len(self.times)
        assert rnk != -1 or at <= self.times[0]
        assert rnk != len(self.times) - 1 or at > self.times[-1]
        assert not (-1 < rnk < len(self.times) - 1) or (self.times[rnk] < at <= self.times[rnk + 1])

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

    @lru_cache
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

        merged = merge_sorted(self.times, other.times)
        times = merged
        if times[0] < new_domain[0]:
            # cut times below new_domain[0]
            rnk = elem_rank(times, new_domain[0])  # => rnk >= 0
            if times[rnk + 1] == new_domain[0]:
                times = times[rnk + 1:]
            else:
                times = [new_domain[0]] + times[rnk + 1:]
        if times[-1] > new_domain[1]:
            # cut times above new_domain[1]
            rnk = elem_rank(times, new_domain[1])  # => rnk <= len(times) - 1
            if times[rnk + 1] == new_domain[1]:
                times = times[:rnk + 2]
            else:
                times = times[:rnk + 1] + [new_domain[1]]
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

    @lru_cache
    def compose(self, f: LinearlyInterpolatedFunction) -> LinearlyInterpolatedFunction:
        g = self
        # We calculate g ⚬ f
        assert f.is_monotone(), "Composition g ⚬ f only implemented for monotone incr. function f"
        assert g.domain[0] <= f.image()[0] and g.domain[1] >= f.image()[1], \
            "The domains do not match for composition!"

        times = []
        values = []
        f_image = f.image()

        f_ind = -1  # Start of analyzed interval
        g_ind = max(0, elem_rank(g.times, f_image[0]))  # Start of interval
        assert g_ind == len(g.times) - 1 or f.domain[0] <= g.times[g_ind + 1]

        while f_ind < len(f.times):
            f_after = f.values[f_ind + 1] if f_ind < len(f.times) - 1 else f_image[1]

            while g_ind < len(g.times) and g.times[g_ind] <= f_after:
                next_time = max(f_image[0], g.times[g_ind])
                if f.gradient(f_ind) != 0:
                    inverse = f.inverse(next_time, f_ind)
                    if len(times) == 0 or inverse > times[-1]:
                        times.append(inverse)
                        values.append(g(next_time))
                g_ind += 1
            if f_ind + 1 < len(f.times):
                if len(times) == 0 or f.times[f_ind + 1] > times[-1]:
                    times.append(f.times[f_ind + 1])
                    values.append(g(f.values[f_ind + 1]))
            f_ind += 1

        return LinearlyInterpolatedFunction(times, values, f.domain)

    def minimum(self, otherf: LinearlyInterpolatedFunction) -> LinearlyInterpolatedFunction:
        # Calculate the pointwise minimum of self and otherf.
        # TODO: This procedure is not perfect yet.
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
            if curr_other_val < curr_min_val - eps:
                # The minimum function has changed!
                # Find the intersecting time with x=next_time:
                # t = x + (g(x) - f(x)) / (grad_f - grad_g)
                grad_min = f[curr_min].gradient(ind[curr_min] - 1)
                grad_other = f[other].gradient(ind[other] - 1)
                difference = grad_min - grad_other
                assert difference > eps
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
            if grad_min > grad_other + eps:
                curr_min_val = f[curr_min](times[-1])
                curr_other_val = f[other](times[-1])
                difference = grad_min - grad_other
                t = times[-1] + (curr_other_val - curr_min_val) / difference
                if times[-1] < t <= new_domain[1]:
                    # Min function will change once again. We need another two points to adjust the gradient.
                    times.append(t)
                    if t < new_domain[1]:
                        times.append(t + 1 if new_domain[1] == float('inf') else new_domain[1])

        values = [min(self(t), otherf(t)) for t in times]
        return LinearlyInterpolatedFunction(times, values, new_domain)

    def is_monotone(self):
        return all(self.values[i] <= self.values[i + 1] for i in range(len(self.values) - 1))

    def image(self) -> Tuple[float, float]:
        assert self.is_monotone(), "Only implemented for monotone functions"
        return self(self.domain[0]), self(self.domain[1])

    def max_t_below_bound(self, bound: float, default: Optional[float] = None):
        """
        Returns max t s.t. self(t) <= bound
        If such a t does not exist, we return default if is given.
        Otherwise we throw an error.
        """
        assert self.is_monotone(), "Only implemented for monotone functions"
        assert default is not None or self(self.domain[0]) <= bound
        if self(self.domain[0]) > bound:
            return default
        index = None
        for index_j in range(len(self.times)):
            if self.values[index_j] > bound:
                index = index_j - 1
                break
        if index is not None:
            return self.inverse(bound, index)
        else:
            if self(self.domain[1]) <= bound:
                return self.domain[1]
            else:
                return self.inverse(bound, len(self.times) - 1)

    def ensure_monotone(self) -> LinearlyInterpolatedFunction:
        """
        This function makes sure that an almost monotone function becomes actually monotone.
        It only fixes values where the monotonicity is broken most likely due to rounding errors.
        """
        new_values = self.values.copy()
        for i in range(len(new_values) - 1):
            assert new_values[i] <= new_values[i + 1] + eps
            new_values[i + 1] = max(new_values[i], new_values[i + 1])
        return LinearlyInterpolatedFunction(self.times, new_values, self.domain)

    def smaller_equals(self, other: LinearlyInterpolatedFunction) -> bool:
        """
        Returns whether self is smaller or equal to other everywhere.
        """
        assert self.domain == other.domain
        assert self.times[0] == self.domain[0] == other.times[0] and self.domain[1] == float('inf')
        f = self
        g = other

        ind_f, ind_g = 0, 0
        if f.values[0] > g.values[0] + eps:
            return False

        while ind_f < len(f.times) - 1 or ind_g < len(g.times) - 1:
            next_time_f = f.times[ind_f + 1] if ind_f < len(f.times) - 1 else float('inf')
            next_time_g = g.times[ind_g + 1] if ind_g < len(g.times) - 1 else float('inf')

            next_time = min(next_time_f, next_time_g)
            if f._eval_with_rank(next_time, ind_f) > g._eval_with_rank(next_time, ind_g) + eps:
                return False
            if next_time_f == next_time:
                ind_f += 1
            if next_time_g == next_time:
                ind_g += 1

        return f.gradient(len(f.times) - 1) <= g.gradient(len(g.times) - 1) + eps

    def extend(self, time: float, value: float):
        assert time >= self.times[-1] - eps
        if time <= self.times[-1]:
            #  Simply replace the last value
            self.values[-1] = value
        else:
            self.times.append(time)
            self.values.append(value)

    def equals(self, other):
        if not isinstance(other, LinearlyInterpolatedFunction):
            return False
        return self.values == other.values and self.times == other.times and self.domain == other.domain

    def integrate(self, start: float, end: float):
        assert self.domain[0] <= start < end <= self.domain[1]
        assert min(self.values) >= 0
        # For two time steps, we integrate by adding (max + min) / 2 * delta_t

        value = 0.
        rnk = elem_lrank(self.times, start)

        if rnk == len(self.times) - 1:
            return (self(start) + self(end)) / 2 * (end - start)

        value += (self.values[rnk + 1] + self(start)) / 2 * (self.times[rnk + 1] - start)
        rnk += 1
        while rnk < len(self.times) - 1 and self.times[rnk + 1] >= end:
            value += (self.values[rnk + 1] + self.values[rnk]) / 2 * (self.times[rnk + 1] - self.times[rnk])
            rnk += 1

        value += (self(end) + self.values[rnk]) / 2 * (end - self.times[rnk])
        return value


identity = LinearlyInterpolatedFunction([0., 1.], [0., 1.])
zero = LinearlyInterpolatedFunction([-1, 0.], [0., 0.])
