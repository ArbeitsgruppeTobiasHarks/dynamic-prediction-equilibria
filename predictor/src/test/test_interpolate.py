import unittest

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
        self.assertListEqual(
            comp.times, [0., 0.5, 1.]
        )
        self.assertListEqual(
            comp.values, [0., 1., 2.]
        )
        plot(comp)

    def test_compose_non_monotone(self):
        g = LinearlyInterpolatedFunction([0., 1., 2.], [0., 2., 0.])
        f = LinearlyInterpolatedFunction([0.5, 1.], [0.5, 1.])
        comp = g.compose(f)
        self.assertListEqual(
            comp.times, [0., 0.5, 1., 2.]
        )
        self.assertListEqual(
            comp.values, [0., 1., 2., 0.]
        )
        plot(comp)

    def test_compose_with_domain(self):
        g = LinearlyInterpolatedFunction([0., 1., 2.], [1., 2., 0.], (float('-inf'), 6.))
        f = LinearlyInterpolatedFunction([-10., -5., -2.], [-5., 0., 6.], (-12., -2.))
        comp = g.compose(f)
        self.assertListEqual(
            comp.times, [-10., -5., -4.5, -4., -2.]
        )
        self.assertListEqual(
            comp.values, [-4., 1., 2., 0., -8.]
        )
        plot(comp)

    def test_sum(self):
        f1 = LinearlyInterpolatedFunction([0, 1, 2], [0., 1., 4.])
        f2 = LinearlyInterpolatedFunction([-1, 4], [-1, 10])

        sum1 = f1.plus(f2)
        plot(sum1)


def plot(f: LinearlyInterpolatedFunction):
    plt.plot([f.domain[0]] + f.times + [f.domain[1]],
             [f(f.domain[0])] + f.values + [f(f.domain[1])])
    plt.grid(which='both', axis='both')
    plt.show()
