import unittest
from typing import List

import matplotlib.pyplot as plt

from utilities.interpolate import LinearlyInterpolatedFunction


class TestLinearlyInterpolatedFunction(unittest.TestCase):
    def test_inverse(self):
        f1 = LinearlyInterpolatedFunction([0., 1.], [1., 3.])
        self.assertEqual(f1.inverse(-1., -1), -1.)
        self.assertEqual(f1.inverse(1., 0), 0.)
        self.assertEqual(f1.inverse(1., -1), 0.)
        self.assertEqual(f1.inverse(2., 0), 0.5)
        self.assertEqual(f1.inverse(3., 0), 1.)
        self.assertEqual(f1.inverse(3., 1), 1.)
        self.assertEqual(f1.inverse(5., 1), 2.)

    def test_compose(self):
        g = LinearlyInterpolatedFunction([0., 1.], [0., 2.])
        f = LinearlyInterpolatedFunction([0.5, 1.], [0.5, 1.])
        comp = g.compose(f)
        self.assertListEqual(comp.times, [0., 0.5, 1.])
        self.assertListEqual(comp.values, [0., 1., 2.])
        plot(comp)

    def test_compose_bounded(self):
        f = LinearlyInterpolatedFunction([0., 1., 5.], [1., 2., 15.], (0, float('inf')))
        g = LinearlyInterpolatedFunction([0., 1.], [0., 1.], (0, float('inf')))
        comp = g.compose(f)
        self.assertListEqual(comp.times, [0, 1, 5])
        self.assertListEqual(comp.values, [1., 2., 15.])
        plot(comp)

    def test_compose_non_monotone(self):
        g = LinearlyInterpolatedFunction([0., 1., 2.], [0., 2., 0.])
        f = LinearlyInterpolatedFunction([0.5, 1.], [0.5, 1.])
        comp = g.compose(f)
        self.assertListEqual(comp.times, [0., 0.5, 1., 2.])
        self.assertListEqual(comp.values, [0., 1., 2., 0.])
        plot(comp)

    def test_compose_with_domain(self):
        g = LinearlyInterpolatedFunction([0., 1., 2.], [1., 2., 0.], (float('-inf'), 6.))
        f = LinearlyInterpolatedFunction([-10., -5., -2.], [-5., 0., 6.], (-12., -2.))
        comp = g.compose(f)
        self.assertListEqual(comp.times, [-10., -5., -4.5, -4., -2.])
        self.assertListEqual(comp.values, [-4., 1., 2., 0., -8.])
        plot(comp)

    def test_another_compose(self):
        g = LinearlyInterpolatedFunction([0., 1.], [0., 1.], (0., float('inf')))
        f = LinearlyInterpolatedFunction([0., 1.], [3., 4.], (0., float('inf')))
        comp = g.compose(f)
        self.assertListEqual(comp.times, [0., 1.])
        self.assertListEqual(comp.values, [3., 4.])
        plot(comp)

    def test_yet_another_compose(self):
        g = LinearlyInterpolatedFunction([7.0, 8.0], [7.0, 8.0], (7.0, float('inf')))
        f = LinearlyInterpolatedFunction([7.0, 8.0, 9.0, 10.0], [23.0, 23.0, 23.0, 24.0], (7.0, float('inf')))
        comp = g.compose(f)
        self.assertListEqual(comp.times, [7., 8., 9., 10.])
        plot(comp)

    def test_sum(self):
        f1 = LinearlyInterpolatedFunction([0, 1, 2], [0., 1., 4.])
        f2 = LinearlyInterpolatedFunction([-1, 4], [-1, 10])

        sum1 = f1.plus(f2)
        plot(sum1)

    def test_sum_with_domains(self):
        f1 = LinearlyInterpolatedFunction([0, 1, 2], [0., 1., 4.], (-0.5, float('inf')))
        f2 = LinearlyInterpolatedFunction([-1, 4], [-1, 10], (float('-inf'), 10.))

        sum1 = f1.plus(f2)
        self.assertListEqual(sum1.times, [-0.5, 0., 1., 2., 4.])
        plot(sum1)

    def test_min(self):
        f1 = LinearlyInterpolatedFunction([0, 1, 2], [0., 0., 0.], (0, float('inf')))
        f2 = LinearlyInterpolatedFunction([0, 1, 2], [1., -1., 1.], (0, float('inf')))
        min = f1.minimum(f2)
        plot_many([f1, f2, min])

    def test_min_inf(self):
        f1 = LinearlyInterpolatedFunction([0., 1., 2.], [0., 1., 4], (0, float('inf')))
        f2 = LinearlyInterpolatedFunction([0., 1., 2.], [0., -1., 3], (0, float('inf')))
        min = f1.minimum(f2)
        plot_many([f1, f2, min])

    def test_another_min(self):
        f = LinearlyInterpolatedFunction(times=[0.4, 1.4, 2.4, 3.4], values=[3.8, 5.2, 6.6, 7.6], domain=(0.4, float('inf')))
        g = LinearlyInterpolatedFunction(times=[0.4, 0.8285714285714285, 1.4, 1.5428571428571427, 2.4, 3.4], values=[3.8, 4.4, 5.2, 5.4, 6.600000000000002, 7.600000000000003], domain=(0.4, float('inf')))
        min = f.minimum(g)


    def test_max_before_bound(self):
        f = LinearlyInterpolatedFunction([0, 2.0, 3.0], [1.0, 1.0, 1.5], (0, float('inf')))
        self.assertEqual(f.max_t_below_bound(0., default=0.), 0.)


def plot(f: LinearlyInterpolatedFunction):
    plt.plot([f.domain[0]] + f.times + [f.domain[1]],
             [f(f.domain[0])] + f.values + [f(f.domain[1])])
    plt.grid(which='both', axis='both')
    plt.show()


def plot_many(fs: List[LinearlyInterpolatedFunction]):
    for f in fs:
        l_time = f.times[0] - (f.times[-1] - f.times[0]) if f.domain[1] == float('-inf') else f.domain[0]
        r_time = f.times[-1] + (f.times[-1] - f.times[0]) if f.domain[1] == float('inf') else f.domain[1]
        plt.plot([l_time] + f.times + [r_time],
                 [f(l_time)] + f.values + [f(r_time)])
    plt.grid(which='both', axis='both')
    plt.show()
