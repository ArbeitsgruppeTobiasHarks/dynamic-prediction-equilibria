from __future__ import annotations

from typing import List, Tuple
from typing import Iterable, List
import cython
from cython.cimports.libc.math import fabs

from cython.cimports.cpython import array
import array


from src.cython_test.array_utils cimport elem_lrank, merge_sorted_many, merge_sorted
from utilities.piecewise_linear import PiecewiseLinear

import cython

cdef double eps = 1e-10

cdef double inf = float("inf")
cdef double neg_inf = float("-inf")


cdef class RightConstant:
    """
    This class defines right-continuous functions with finitely many break points (xᵢ, yᵢ).
    The breakpoints are encoded in the two lists times = [x₀, ..., xₙ] and values = [y₀,..., yₙ]
    where we assume that times is strictly increasing.
    This encodes the function f(x) = yᵢ with i maximal s.t. xᵢ <= x (or i = 0 if x₀ > x).
    """

    cdef public array.array times
    cdef public array.array values
    cdef public (double, double) domain

    def __init__(
        self,
        times: array.array,
        values: array.array,
        domain: cython.ctuple[cython.double, cython.double] = (neg_inf, inf),
    ):
        assert times.typecode == "d" == values.typecode

        self.times = times
        self.values = values
        self.domain = domain
        assert len(self.values) == len(self.times) >= 1
        assert all(
            float("-inf") < self.values[i] < float("inf")
            for i in range(len(self.times))
        )
        assert all(
            self.domain[0] <= self.times[i] <= self.domain[1]
            for i in range(len(self.times))
        )

    def __json__(self):
        return {
            "times": self.times,
            "values": self.values,
            "domain": [
                "-Infinity" if self.domain[0] == float("-inf") else self.domain[0],
                "Infinity" if self.domain[1] == float("inf") else self.domain[1],
            ],
        }
    
    def __get_state__(self):
        array.resize(self.times, len(self.times))
        array.resize(self.values, len(self.values))
        return __dict__(self)

    def __call__(self, at: cython.double) -> cython.double:
        return self.eval(at)

    @cython.ccall
    def eval(self, at: cython.double) -> cython.double:
        assert self.domain[0] <= at <= self.domain[1], f"Function not defined at {at}."
        rnk = elem_lrank(self.times, at)
        return self._eval_with_lrank(at, rnk)

    @cython.cfunc
    def _eval_with_lrank(self, at: cython.double, rnk: cython.Py_ssize_t) -> cython.double:
        assert self.domain[0] <= at <= self.domain[1], f"Function not defined at {at}."
        assert -1 <= rnk <= len(self.times)
        assert rnk == elem_lrank(self.times, at)

        if rnk == -1:
            return self.values.data.as_doubles[0]
        else:
            return self.values.data.as_doubles[rnk]

    @cython.ccall
    def extend(self, start_time: cython.double, value: cython.double):
        assert start_time >= self.times[-1] - eps
        cdef double* values_p = self.values.data.as_doubles
        cdef double* times_p = self.times.data.as_doubles
    
        last_index: cython.Py_ssize_t = self.times.ob_size - 1
        last_value: cython.double = values_p[last_index]
        if fabs(last_value - value) <= eps:
            return
        if fabs(start_time - times_p[last_index]) <= eps:
            #  Simply replace the last value
            values_p[last_index] = value
        else:
            array.resize_smart(self.times, last_index + 2)
            array.resize_smart(self.values, last_index + 2)
            self.times.data.as_doubles[last_index + 1] = start_time
            self.values.data.as_doubles[last_index + 1] = value

    def equals(self, other):
        if not isinstance(other, RightConstant):
            return False
        return (
            self.values == other.values
            and self.times == other.times
            and self.domain == other.domain
        )

    @cython.cfunc
    def plus(self, other: RightConstant) -> RightConstant:
        assert self.domain == other.domain

        self_times_p = self.times.data.as_doubles
        self_values_p = self.values.data.as_doubles
        other_times_p = other.times.data.as_doubles
        other_values_p = other.values.data.as_doubles

        new_times: array.array = merge_sorted(self.times, other.times)
        new_values: array.array = array.clone(new_times, new_times.ob_size, zero=True)
        new_values_p = new_values.data.as_doubles

        lptr: cython.Py_ssize_t = 0
        rptr: cython.Py_ssize_t = 0
        for i in range(new_times.ob_size):
            time: cython.double = new_times.data.as_doubles[i]

            while lptr < self.times.ob_size - 1 and self_times_p[lptr + 1] <= time:
                lptr += 1
            while rptr < other.times.ob_size - 1 and other_times_p[rptr + 1] <= time:
                rptr += 1
            new_values_p[i] = self_values_p[lptr] + other_values_p[rptr]

        return RightConstant(new_times, new_values, self.domain)

    def __radd__(self, other):
        if other == 0:
            return self
        if not isinstance(other, RightConstant):
            raise TypeError("Can only add a RightConstantFunction.")
        assert self.domain == other.domain
        return self.plus(other)

    @staticmethod
    def sum(functions: List[RightConstant], domain=(0, float("inf"))) -> RightConstant:
        if len(functions) == 0:
            return RightConstant(array.array("d", [0.0]), array.array("d", [0.0]), domain)
        new_times: array.array = merge_sorted_many([f.times for f in functions])
        new_values: array.array = array.clone(new_times, len(new_times), zero=True)
        
        ptrs_template: array.array = array.array("l", [])
        ptrs: array.array() = array.clone(ptrs_template, len(functions), zero=True)

        for i, time in enumerate(new_times):
            for j in range(len(ptrs)):
                while (
                    ptrs[j] < len(functions[j].times) - 1
                    and functions[j].times[ptrs[j] + 1] <= time
                ):
                    ptrs[j] += 1
            new_values[i] = sum(
                functions[j].values[ptrs[j]] for j in range(len(functions))
            )
        return RightConstant(new_times, new_values, domain)

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
        new_times: array.array = array.clone(self.times, len(self.times), zero=False)
        new_values: array.array = array.clone(self.times, len(self.times), zero=False)
        new_times.data.as_doubles[0] = self.times.data.as_doubles[0]
        new_values.data.as_doubles[0] = self.values.data.as_doubles[0]
        size: cython.Py_ssize_t = 1

        for i in range(0, len(self.times) - 1):
            # Add i+1, if it's necessary.
            if fabs(self.values.data.as_doubles[i] - self.values.data.as_doubles[i + 1]) >= 1000 * eps:
                new_times.data.as_doubles[size] = self.times.data.as_doubles[i + 1]
                new_values.data.as_doubles[size] = self.values.data.as_doubles[i + 1]
                size += 1

        array.resize_smart(new_times, size)
        array.resize_smart(new_values, size)

        return RightConstant(new_times, new_values, self.domain)

    def integral(self) -> PiecewiseLinear:
        """
        Returns the integral starting from self.times[0] to x of self.
        """
        assert self.times[0] == self.domain[0] and self.domain[1] >= self.times[len(self.times) - 1] + 1
        times = self.times
        values = [0.0] * len(times)
        for i in range(len(times) - 1):
            values[i + 1] = values[i] + self.values[i] * (times[i + 1] - times[i])
        return PiecewiseLinear(
            times, values, self.values[0], self.values[-1], self.domain
        )
