from test.test_interpolate import plot as plot_pw_linear
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison

from core.machine_precision import eps
from utilities.right_constant import RightConstant


@image_comparison(baseline_images=["integral"], extensions=["pdf"])
def test_integral():
    times = [
        0.0,
        3.0,
        3.25,
        5.625,
        6.54375,
        6.89375,
        7.2,
        7.231249999999999,
        8.018749999999999,
        83.35624999999997,
    ]
    values = [
        0.0,
        0.3416666666666667,
        0.45238095238095244,
        0.4285714285714286,
        0.41071428571428575,
        0.4285714285714286,
        0.41071428571428575,
        0.45833333333333337,
        0.41071428571428575,
        0.3472222222222222,
    ]
    f = RightConstant(times, values, domain=(0.0, float("inf")))
    integral = f.integral()
    plot_pw_linear(integral)


def plot_many(fs: List[RightConstant]) -> Figure:
    max_times = max(f.times[-1] for f in fs)
    min_times = min(f.times[0] for f in fs)

    fig, ax = plt.subplots()
    for i, f in enumerate(fs):
        left = max(f.domain[0], min_times - 1)
        right = min(f.domain[1], max_times + 1)

        ax.plot(
            [left]
            + [max(x, f.domain[0]) for t in f.times for x in [t - eps, t]]
            + [right],
            [f(left)]
            + [f(max(x, f.domain[0])) for t in f.times for x in [t - eps, t]]
            + [f(right)],
            label=str(i),
        )
    ax.grid(which="both", axis="both")
    ax.legend()
    return fig
