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



def test_sum():
    f1 = LinearlyInterpolatedFunction([0, 1, 2], [0., 1., 4.])
    f2 = LinearlyInterpolatedFunction([-1, 4], [-1, 10])

    sum1 = f1.plus(f2)
    plt.plot(f1.times, f1.values)
    plt.plot(f2.times, f2.values)
    plt.plot(sum1.times, sum1.values)
    plt.show()


def test_compose():
    g = LinearlyInterpolatedFunction([0., 0.5, 1], [0., 2., 3.], (0., 5.))
    f = LinearlyInterpolatedFunction([0.5, 1.], [0.5, 1.], (0., 5.))
    comp = g.compose(f)
    plt.plot([comp.domain[0]] + comp.times + [comp.domain[1]],
             [comp(comp.domain[0])] + comp.values + [comp(comp.domain[1])])
    plt.show()


test_compose()
