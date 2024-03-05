import unittest
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison

from utilities.piecewise_linear import PiecewiseLinear


def test_inverse():
    f = PiecewiseLinear([0.0, 1.0], [1.0, 3.0], 4.0, 1.0)
    assert f.inverse(-1.0, -1) == -0.5
    assert f.inverse(1.0, 0) == 0.0
    assert f.inverse(1.0, -1) == 0.0
    assert f.inverse(2.0, 0) == 0.5
    assert f.inverse(3.0, 0) == 1.0
    assert f.inverse(3.0, 1) == 1.0
    assert f.inverse(5.0, 1) == 3.0


@image_comparison(baseline_images=["compose"], extensions=["pdf"])
def test_compose():
    g = PiecewiseLinear([0.0, 1.0], [0.0, 2.0], 0.5, 1.0)
    f = PiecewiseLinear([0.5, 1.0], [0.5, 1.0], 1.0, 1.0)
    comp = g.compose(f)
    assert comp.times, [0.0, 0.5, 1.0]
    assert comp.values, [0.0, 1.0, 2.0]
    plot(comp)


@image_comparison(baseline_images=["compose_bounded"], extensions=["pdf"])
def test_compose_bounded():
    f = PiecewiseLinear([0.0, 1.0, 5.0], [1.0, 2.0, 15.0], 0.0, 2.0, (0, float("inf")))
    g = PiecewiseLinear([0.0, 1.0], [0.0, 1.0], 0.0, 1.0, (0, float("inf")))
    comp = g.compose(f)
    plot(comp)
    assert comp.times, [0, 1, 5]
    assert comp.values, [1.0, 2.0, 15.0]


@image_comparison(baseline_images=["compose_non_monotone"], extensions=["pdf"])
def test_compose_non_monotone():
    g = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, 2.0, 0.0], 2, -2)
    f = PiecewiseLinear([0.5, 1.0], [0.5, 1.0], 1.0, 1.0)
    comp = g.compose(f)
    assert comp.times, [0.0, 0.5, 1.0, 2.0]
    assert comp.values, [0.0, 1.0, 2.0, 0.0]
    plot(comp)


@image_comparison(baseline_images=["compose_with_domain"], extensions=["pdf"])
def test_compose_with_domain():
    g = PiecewiseLinear([0.0, 1.0, 2.0], [1.0, 2.0, 0.0], 1, -2, (float("-inf"), 6.0))
    f = PiecewiseLinear([-10.0, -5.0, -2.0], [-5.0, 0.0, 6.0], 1, 2, (-12.0, -2.0))
    comp = g.compose(f)
    assert comp.times == [-10.0, -5.0, -4.5, -4.0, -2.0]
    assert comp.values == [-4.0, 1.0, 2.0, 0.0, -8.0]
    plot(comp)


@image_comparison(baseline_images=["another_compose"], extensions=["pdf"])
def test_another_compose():
    g = PiecewiseLinear([0.0, 1.0], [0.0, 1.0], 0.0, 1.0, (0.0, float("inf")))
    f = PiecewiseLinear([0.0, 1.0], [3.0, 4.0], 0.0, 1.0, (0.0, float("inf")))
    comp = g.compose(f)
    assert comp.times == [0.0, 1.0]
    assert comp.values == [3.0, 4.0]
    plot(comp)


@image_comparison(baseline_images=["yet_another_compose"], extensions=["pdf"])
def test_yet_another_compose():
    g = PiecewiseLinear([7.0, 8.0], [7.0, 8.0], 1.0, 1.0, (7.0, float("inf")))
    f = PiecewiseLinear(
        [7.0, 8.0, 9.0, 10.0],
        [23.0, 23.0, 23.0, 24.0],
        0.0,
        1.0,
        (7.0, float("inf")),
    )
    comp = g.compose(f)
    assert comp.times == [7.0, 8.0, 9.0, 10.0]
    plot(comp)


@image_comparison(baseline_images=["sum"], extensions=["pdf"])
def test_sum():
    f1 = PiecewiseLinear([0, 1, 2], [0.0, 1.0, 4.0], 1.0, 3.0)
    f2 = PiecewiseLinear([-1, 4], [-1, 10], 11.0 / 5, 11.0 / 5)

    sum1 = f1.plus(f2)
    plot(sum1)


@image_comparison(baseline_images=["sum_with_domains"], extensions=["pdf"])
def test_sum_with_domains():
    f1 = PiecewiseLinear([0, 1, 2], [0.0, 1.0, 4.0], 1.0, 3.0, (-0.5, float("inf")))
    f2 = PiecewiseLinear([-1, 4], [-1, 10], 11.0 / 5, 11.0 / 5, (float("-inf"), 10.0))

    sum1 = f1.plus(f2)
    assert sum1.times == [-0.5, 0.0, 1.0, 2.0, 4.0]
    plot(sum1)


@image_comparison(baseline_images=["min"], extensions=["pdf"])
def test_min():
    f1 = PiecewiseLinear([0], [0.0], 0.0, 0.0, (0, float("inf")))
    f2 = PiecewiseLinear([0, 1, 2], [1.0, -1.0, 1.0], -2, 2, (0, float("inf")))
    min = f1.minimum(f2)
    plot_many([f1, f2, min])


@image_comparison(baseline_images=["min_inf"], extensions=["pdf"])
def test_min_inf():
    f1 = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, 1.0, 4], 1, 3.0, (0, float("inf")))
    f2 = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, -1.0, 3], -1, 4, (0, float("inf")))
    min = f1.minimum(f2)
    plot_many([f1, f2, min])


@image_comparison(baseline_images=["another_min"], extensions=["pdf"])
def test_another_min():
    f = PiecewiseLinear(
        [0.4, 1.4, 2.4, 3.4],
        [3.8, 5.2, 6.6, 7.6],
        1.4,
        1.0,
        domain=(0.4, float("inf")),
    )
    g = PiecewiseLinear(
        [0.4, 0.8285714285714285, 1.4, 1.5428571428571427, 2.4, 3.4],
        [3.8, 4.4, 5.2, 5.4, 6.600000000000002, 7.600000000000003],
        1.4000000000000001,
        1.0000000000000009,
        domain=(0.4, float("inf")),
    )
    min = f.minimum(g)
    plot_many([f, g, min])


@image_comparison(baseline_images=["advanced_min"], extensions=["pdf"])
def test_advanced_min():
    f = PiecewiseLinear(
        times=[
            0.0,
            0.015969092810812158,
            0.04629633380971987,
            0.12643816663307483,
            0.1285371270785789,
            0.144300592240585,
            0.19025182845019595,
            0.22959474304073135,
            0.2806477323102734,
            0.2930668960294671,
            0.3554256212461928,
            0.3930433604684669,
            0.4079817990419251,
            0.4440941314775547,
            0.535437576499388,
            0.5359476986548581,
            0.5678608390044512,
            0.5703827899549732,
            0.5752213677464768,
            0.6443465760784124,
            0.6804293119962342,
            0.6871161397960552,
            0.7552800801879466,
            0.8148869897117921,
            0.8846421250080317,
            0.9179026576424831,
            0.9275270169850957,
            1.0,
            1.0355082366493038,
            1.038643025440888,
            1.069530143124493,
            1.1512498238436542,
            1.1561459970658885,
            1.197299992294572,
            1.1981361378239939,
            1.2705997757744059,
            1.3346882991710227,
            1.3820318454856277,
            1.4447407109483028,
            1.4534302325581394,
            1.5501128190268112,
            1.583358951477351,
            1.6632351957152753,
            1.66402084873448,
            1.682088675103044,
            1.718939482648063,
            1.7867455094316362,
            1.8563700200699422,
            1.9761139745699892,
            2.0,
            2.032786468797299,
            2.063843495298356,
            2.0910927095143608,
            2.214731445105903,
            2.2753614799558095,
            2.2764233019321654,
            2.349118662934413,
            2.3665992147387493,
            2.4248047482090014,
            2.4288227501820616,
            2.682627199744312,
            2.7350244702891047,
            2.827489018989151,
            2.8524299432705154,
            2.901390847504329,
            2.9026877474278314,
            3.0,
            3.067436279420181,
            3.1701008458807856,
            3.2719307228915664,
            3.3708000348501455,
            3.4347975953400387,
            3.619195932490292,
            3.620627277705866,
            3.6717576670317627,
            3.8157469879518082,
            3.940723986856517,
            4.0,
            4.176609090909091,
            4.239715757575757,
            4.46419,
            4.465932424242425,
            4.528175151515152,
            4.825831212121212,
            5.0,
            5.09427,
            5.163687333333334,
            5.410609000000001,
            5.412525666666667,
            5.480992666666666,
            5.808414333333333,
            6.0,
            6.09427,
            6.163687333333333,
            6.412525666666666,
            6.480992666666666,
            6.808414333333333,
            7.0,
            7.09427,
            7.163687333333334,
            7.480992666666667,
            7.808414333333333,
            8.0,
            8.09427,
            8.480992666666667,
            9.0,
            9.09427,
            10.0,
        ],
        values=[
            61.729694366663544,
            61.76936866666646,
            61.84471496666646,
            62.04262333333312,
            62.04780666666645,
            62.086734172231445,
            62.20020966666645,
            62.28853366666645,
            62.40314653333312,
            62.43085933333312,
            62.570009997521176,
            62.653952274722,
            62.68772533333311,
            62.769368666666445,
            62.975879374414994,
            62.977032666666446,
            63.04262333333311,
            63.047806666666446,
            63.057751333333115,
            63.200209666666446,
            63.27457163308941,
            63.288533666666446,
            63.43085933333311,
            63.55531799999977,
            63.687725333333105,
            63.750859587358576,
            63.76936866666644,
            63.90874502851537,
            63.977032666666446,
            63.98306133333311,
            64.04262333333311,
            64.20020966666644,
            64.20965133333311,
            64.28853366666644,
            64.29013635700065,
            64.4308593333331,
            64.55531799999977,
            64.64725824666644,
            64.76936866666644,
            64.7862894182308,
            64.97703266666645,
            65.04262333333311,
            65.20020966666645,
            65.20175966666645,
            65.237190607022,
            65.30965133333312,
            65.43085933333312,
            65.55531799999979,
            65.76936866666645,
            65.81206660234645,
            65.87144586023683,
            65.92784583333312,
            65.97703266666646,
            66.20020966666647,
            66.30965133333314,
            66.3115679999998,
            66.43085933333313,
            66.45954449705461,
            66.5553179999998,
            66.56192936666646,
            66.97703266666647,
            67.06272966666647,
            67.20020966666648,
            67.23729283333314,
            67.30965133333315,
            67.31156799999982,
            67.45538414755538,
            67.55531799999983,
            67.7074566666665,
            67.8446400666665,
            67.9770326666665,
            68.0627296666665,
            68.30965133333318,
            68.31156799999985,
            68.38003499999985,
            68.55531799999986,
            68.70745666666653,
            68.7796153333332,
            68.99331233333321,
            69.06272966666654,
            69.30965133333322,
            69.3115679999999,
            69.38003499999989,
            69.70745666666656,
            69.89904233333323,
            69.99331233333324,
            70.06272966666657,
            70.30965133333325,
            70.31156799999992,
            70.38003499999992,
            70.70745666666659,
            70.89904233333326,
            70.99331233333326,
            71.0627296666666,
            71.31156799999994,
            71.38003499999994,
            71.7074566666666,
            71.89904233333327,
            71.99331233333328,
            72.06272966666661,
            72.38003499999995,
            72.70745666666662,
            72.89904233333328,
            72.99331233333328,
            73.38003499999995,
            73.89904233333328,
            73.99331233333328,
            74.89904233333327,
        ],
        first_slope=0.0,
        last_slope=1.0,
        domain=(0.0, float("inf")),
    )
    g = PiecewiseLinear(
        times=[
            0.0,
            0.0016084519928005747,
            0.034470278598143445,
            0.06479751959705116,
            0.144939352420406,
            0.14703831286591018,
            0.1628017780279163,
            0.20875301423752723,
            0.24809592882806264,
            0.2991489180976046,
            0.3115680818167983,
            0.373926807033524,
            0.41154454625579817,
            0.4264829848292564,
            0.462595317264886,
            0.5544488844421894,
            0.5863620247917823,
            0.5888839757423046,
            0.593722553533808,
            0.6628477618657438,
            0.6989304977835655,
            0.7056173255833864,
            0.7737812659752779,
            0.7979394124847001,
            0.8333881754991233,
            0.903143310795363,
            0.9364038434298146,
            0.9460282027724269,
            1.0,
            1.0540094224366352,
            1.0571442112282194,
            1.0880313289118242,
            1.1697510096309853,
            1.17464718285322,
            1.2158011780819031,
            1.2166373236113255,
            1.289100961561737,
            1.353189484958354,
            1.4005330312729587,
            1.4632418967356342,
            1.569886144109783,
            1.603569725671514,
            1.6844969731230952,
            1.6852929636820262,
            1.7035985246080712,
            1.7159320685434516,
            1.7406095080483548,
            1.8084155348319282,
            1.8780400454702348,
            1.9977839999702813,
            2.0,
            2.054175065296289,
            2.085232091797346,
            2.112481306013351,
            2.2361200416048925,
            2.2967500764547992,
            2.2978118984311555,
            2.3705072594334027,
            2.446351559688731,
            2.4503804834430545,
            2.625691935971006,
            2.7046601748946872,
            2.7570574454394796,
            2.8495219941395264,
            2.8744629184208894,
            2.9234238226547036,
            2.924720722578206,
            3.0,
            3.0894095266258708,
            3.192074093086475,
            3.392177683791024,
            3.4557897168080864,
            3.530321084337349,
            3.639736348700587,
            3.6411676939161612,
            3.6922980832420587,
            3.8362874041621033,
            3.9612644030668127,
            4.0,
            4.254643691460055,
            4.4365654545454545,
            4.460925757575758,
            4.462668181818183,
            4.5249109090909085,
            4.82256696969697,
            5.0,
            5.160096666666667,
            5.380222,
            5.407018333333334,
            5.4089350000000005,
            5.477402,
            5.804823666666667,
            6.0,
            6.160096666666666,
            6.380222,
            6.408935,
            6.477402,
            6.804823666666666,
            7.0,
            7.160096666666666,
            7.380222,
            7.477402,
            7.804823666666666,
            8.0,
            8.380222,
            8.477402,
            9.0,
            9.380222,
            10.0,
        ],
        values=[
            61.683729226124115,
            61.68772533333313,
            61.76936866666646,
            61.84471496666646,
            62.04262333333312,
            62.04780666666645,
            62.086734172231445,
            62.20020966666645,
            62.28853366666645,
            62.40314653333312,
            62.43085933333312,
            62.570009997521176,
            62.653952274722,
            62.68772533333311,
            62.769368666666445,
            62.977032666666446,
            63.04262333333311,
            63.047806666666446,
            63.057751333333115,
            63.200209666666446,
            63.27457163308941,
            63.288533666666446,
            63.43085933333311,
            63.48130131611193,
            63.55531799999977,
            63.687725333333105,
            63.750859587358576,
            63.76936866666644,
            63.87316448722043,
            63.977032666666446,
            63.98306133333311,
            64.04262333333311,
            64.20020966666644,
            64.20965133333311,
            64.28853366666644,
            64.29013635700065,
            64.4308593333331,
            64.55531799999977,
            64.64725824666644,
            64.76936866666644,
            64.97703266666645,
            65.04262333333311,
            65.20020966666645,
            65.20175966666645,
            65.237190607022,
            65.26112742546924,
            65.30965133333312,
            65.43085933333312,
            65.55531799999979,
            65.76936866666645,
            65.77332992120891,
            65.87144586023683,
            65.92784583333312,
            65.97703266666646,
            66.20020966666647,
            66.30965133333314,
            66.3115679999998,
            66.43085933333313,
            66.5553179999998,
            66.56192936666646,
            66.8478782068887,
            66.97703266666647,
            67.06272966666647,
            67.20020966666648,
            67.23729283333314,
            67.30965133333315,
            67.31156799999982,
            67.42282198275538,
            67.55531799999983,
            67.7074566666665,
            67.9770326666665,
            68.0627296666665,
            68.16313699999984,
            68.30965133333318,
            68.31156799999985,
            68.38003499999985,
            68.55531799999986,
            68.70745666666653,
            68.75461079999987,
            69.06272966666654,
            69.28285499999988,
            69.30965133333322,
            69.3115679999999,
            69.38003499999989,
            69.70745666666656,
            69.9026329999999,
            70.06272966666657,
            70.28285499999991,
            70.30965133333325,
            70.31156799999992,
            70.38003499999992,
            70.70745666666659,
            70.90263299999992,
            71.0627296666666,
            71.28285499999994,
            71.31156799999994,
            71.38003499999994,
            71.7074566666666,
            71.90263299999994,
            72.06272966666661,
            72.28285499999996,
            72.38003499999995,
            72.70745666666662,
            72.90263299999995,
            73.28285499999996,
            73.38003499999995,
            73.90263299999995,
            74.28285499999996,
            74.90263299999997,
        ],
        first_slope=0.0,
        last_slope=1.0,
        domain=(0.0, float("inf")),
    )
    min = f.minimum(g)
    plot_many([f, g, min])


def test_max_before_bound():
    f = PiecewiseLinear([0, 2.0, 3.0], [1.0, 1.0, 1.5], 0.0, 3.0, (0, float("inf")))
    assert f.max_t_below(0.0, default=0.0) == 0.0


def plot(f: PiecewiseLinear) -> Figure:
    left = max(f.domain[0], f.times[0] - 1)
    right = min(f.domain[1], f.times[-1] + 1)
    fig, ax = plt.subplots()
    ax.plot([left] + f.times + [right], [f(left)] + f.values + [f(right)])
    ax.grid(which="both", axis="both")
    return fig


def plot_many(fs: List[PiecewiseLinear]) -> Figure:
    max_times = max(f.times[-1] for f in fs)
    min_times = min(f.times[0] for f in fs)
    fig, ax = plt.subplots()
    for i, f in enumerate(fs):
        left = max(f.domain[0], min_times - 1)
        right = min(f.domain[1], max_times + 1)
        ax.plot(
            [left] + f.times + [right], [f(left)] + f.values + [f(right)], label=str(i)
        )
    ax.grid(which="both", axis="both")
    ax.legend()
    return fig
