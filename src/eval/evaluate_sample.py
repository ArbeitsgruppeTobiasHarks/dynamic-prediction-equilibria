import json
from test.sample_network import build_sample_network
from typing import Dict

from core.network import Network
from core.predictor import Predictor
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.linear_regression_predictor import LinearRegressionPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate import COLORS, evaluate_single_run
from utilities.json_encoder import JSONEncoder
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def build_predictors(network: Network) -> Dict[PredictorType, Predictor]:
    prediction_horizon = 10.0
    return {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, prediction_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(
            network, prediction_horizon, delta=5.0
        ),
        PredictorType.MACHINE_LEARNING_SK_FULL_NET: LinearRegressionPredictor(network),
    }


def eval_sample():
    max_demand = 30.0
    step_size = 0.25
    inflow_horizon = 25.0
    reroute_interval = 0.25
    horizon = 500.0
    demand = 0.0 + step_size
    avg_times = [[], [], [], [], [], []]
    comp_times = []
    while demand <= max_demand:
        network = build_sample_network()
        net_inflow = RightConstant(
            [0.0, inflow_horizon], [demand, 0.0], (0, float("inf"))
        )
        network.add_commodity({0: net_inflow}, 2, PredictorType.ZERO)
        times, comp_time, flow = evaluate_single_run(
            network,
            flow_id=None,
            focused_commodity_index=0,
            split=True,
            horizon=horizon,
            future_timesteps=1,
            prediction_interval=1.0,
            reroute_interval=reroute_interval,
            suppress_log=True,
            inflow_horizon=inflow_horizon,
            json_eval_path=None,
            build_predictors=build_predictors,
        )
        for i, val in enumerate(times):
            avg_times[i].append(val)
        comp_times.append(comp_time)
        print(f"Calculated for demand={demand}. times={times}")
        demand += step_size
    print()
    print(avg_times)
    avg_comp_time = sum(comp_times) / len(comp_times)
    print()
    print(f"Average Computation Time: {avg_comp_time}")

    with open("./avg_times_sample.json", "w") as file:
        JSONEncoder().dump(avg_times, file)

    print("Successfully saved these travel times in ./avg_times_sample.json")
    print()
    sample_from_file_to_tikz()
    sample_regrets_from_file_to_tikz()
    print("A tikz diagram was saved to ./avg_times_sample.tikz.")


def sample_from_file_to_tikz():
    configs = [
        {"label": "$\\hat q^{\\text{Z}}$", "color": "blue"},
        {"label": "$\\hat q^{\\text{C}}$", "color": "red"},
        {
            "label": "$\\hat q^{\\text{L}}$",
            "color": "{rgb,255:red,0; green,128; blue,0}",
        },
        {"label": "$\\hat q^{\\text{RL}}$", "color": "orange"},
        {"label": "$\\hat q^{\\text{ML}}$", "color": "black"},
        {"label": "$\\text{OPT}$", "color": "purple", "dashed": True},
    ]
    with open("./avg_times_sample.json", "r") as file:
        avg_times = json.load(file)
    tikz = """\\begin{tikzpicture}
    \\begin{axis}[
        xlabel={Total Inflow $\\sum_i \\bar u_i$},
        ylabel={Average Travel Time $T^{\\text{avg}}_i$},
        legend entries={
    """
    for c in configs:
        tikz += c["label"] + ",\n"
    tikz += """},
        legend pos=north west,
    ]
    """

    for c, values in enumerate(avg_times):
        tikz += "\n\\addplot[color=" + configs[c]["color"]
        if "dashed" in configs[c]:
            tikz += ", dashed"
        tikz += "]\n  coordinates { \n"

        for i, y in enumerate(values):
            x = (i + 1) * 0.25
            tikz += f"({x}, {y})"

        tikz += "\n};\n"

    tikz += "\\end{axis}\n\\end{tikzpicture}"

    with open("./avg_times_sample.tikz", "w") as file:
        file.write(tikz)


def sample_regrets_from_file_to_tikz():
    configs = [
        {"label": "$\\hat q^{\\text{Z}}$", "color": "blue"},
        {"label": "$\\hat q^{\\text{C}}$", "color": "red"},
        {
            "label": "$\\hat q^{\\text{L}}$",
            "color": "{rgb,255:red,0; green,128; blue,0}",
        },
        {"label": "$\\hat q^{\\text{RL}}$", "color": "orange"},
        {"label": "$\\hat q^{\\text{ML}}$", "color": "black"},
    ]
    with open("./avg_times_sample.json", "r") as file:
        avg_times = json.load(file)
    tikz = """\\begin{tikzpicture}
    \\begin{axis}[
        xlabel={Total Inflow $\\sum_i \\bar u_i$},
        ylabel={Regret $T^{\\text{avg}}_i / T^{\\text{avg}}_{i,\\text{OPT}}$},
        legend entries={
    """
    for c in configs:
        tikz += c["label"] + ",\n"
    tikz += """},
        legend pos=south east,
    ]
    """

    for c, values in enumerate(avg_times):
        if c == len(avg_times) - 1:
            break
        tikz += "\n\\addplot[color=" + configs[c]["color"]
        if "dashed" in configs[c]:
            tikz += ", dashed"
        tikz += "]\n  coordinates { \n"

        for i, y in enumerate(values):
            x = (i + 1) * 0.25
            tikz += f"({x}, {y / avg_times[-1][i]})"

        tikz += "\n};\n"

    tikz += "\\end{axis}\n\\end{tikzpicture}"

    with open("./regrets_sample.tikz", "w") as file:
        file.write(tikz)
