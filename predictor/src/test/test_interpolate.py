import matplotlib.pyplot as plt

from utilities.interpolate import LinearlyInterpolatedFunction


def test_sum():
    f1 = LinearlyInterpolatedFunction([0, 1, 2], [0., 1., 4.])
    f2 = LinearlyInterpolatedFunction([-1, 4], [-1, 10])

    sum1 = f1.plus(f2)
    plt.plot(f1.times, f1.values)
    plt.plot(f2.times, f2.values)
    plt.plot(sum1.times, sum1.values)
    plt.show()


def test_compose():
    g = LinearlyInterpolatedFunction([0., 0.5, 1], [0., 2., 3.], (0., 1.))
    f = LinearlyInterpolatedFunction([0., 1.], [0., 1.], (0., 1.))
    comp = g.compose(f)
    plt.plot(comp.times, comp.values)
    plt.show()

test_compose()
