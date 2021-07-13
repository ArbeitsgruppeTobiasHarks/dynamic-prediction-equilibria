import unittest
from typing import List

import matplotlib.pyplot as plt

from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant


class TestRightConstantFunction(unittest.TestCase):
    def test_integral(self):
        times = [0.0, 3.0, 3.25, 5.625, 6.54375, 6.89375, 7.2, 7.231249999999999, 8.018749999999999, 83.35624999999997]
        values = [0.0, 0.3416666666666667, 0.45238095238095244, 0.4285714285714286, 0.41071428571428575, 0.4285714285714286, 0.41071428571428575, 0.45833333333333337, 0.41071428571428575, 0.3472222222222222]
        f = RightConstant(times, values, domain=(0., float('inf')))
        integral = f.integral()
        plot(integral)


def plot(f: PiecewiseLinear):
    plt.plot([f.domain[0]] + f.times + [f.domain[1]],
             [f(f.domain[0])] + f.values + [f(f.domain[1])])
    plt.grid(which='both', axis='both')
    plt.show()


def plot_many(fs: List[PiecewiseLinear]):
    for f in fs:
        l_time = f.times[0] - (f.times[-1] - f.times[0]) if f.domain[1] == float('-inf') else f.domain[0]
        r_time = f.times[-1] + (f.times[-1] - f.times[0]) if f.domain[1] == float('inf') else f.domain[1]
        plt.plot([l_time] + f.times + [r_time],
                 [f(l_time)] + f.values + [f(r_time)])
    plt.grid(which='both', axis='both')
    plt.show()
