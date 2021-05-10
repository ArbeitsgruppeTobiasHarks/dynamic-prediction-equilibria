import unittest
from typing import Dict, Any

import matplotlib.pyplot as plt

from core.time_refinement import time_refinement
from test.sample_network import build_sample_network
from utilities.interpolate import LinearlyInterpolatedFunction


class TestTimeRefinement(unittest.TestCase):
    def test_time_refinement(self):
        network = build_sample_network()
        weights = [
            LinearlyInterpolatedFunction([0., 1.], [1., 1.]),
            LinearlyInterpolatedFunction([0., 1., 5.], [1., 1., 10.]),
            LinearlyInterpolatedFunction([0., 1.], [1., 1.]),
            LinearlyInterpolatedFunction([0., 1.], [1., 1.]),
            LinearlyInterpolatedFunction([0., 1.], [1., 1.])
        ]
        g, tau = time_refinement(network.graph, network.graph.nodes[0], weights, 0)
        plot_many(g)


def plot(f: LinearlyInterpolatedFunction):
    plt.plot([f.domain[0]] + f.times + [f.domain[1]],
             [f(f.domain[0])] + f.values + [f(f.domain[1])])
    plt.grid(which='both', axis='both')
    plt.show()


def plot_many(fs: Dict[Any, LinearlyInterpolatedFunction]):
    for key in fs:
        f = fs[key]
        l_time = f.times[0] - (f.times[-1] - f.times[0]) if f.domain[1] == float('-inf') else f.domain[0]
        r_time = f.times[-1] + (f.times[-1] - f.times[0]) if f.domain[1] == float('inf') else f.domain[1]
        plt.plot([l_time] + f.times + [r_time],
                 [f(l_time)] + f.values + [f(r_time)], label=key)
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()
