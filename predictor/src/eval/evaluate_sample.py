import json

from core.predictors.predictor_type import PredictorType
from eval.evaluate import evaluate_single_run
from eval.sample_network import build_sample_network
from utilities.right_constant import RightConstant


def eval_sample():
    max_demand = 30.
    step_size = 0.25
    inflow_horizon = 25.
    reroute_interval = 0.25
    horizon = 100.
    demand = 0. + step_size
    avg_times = [[], [], [], [], [], []]
    while demand < max_demand:
        network = build_sample_network()
        net_inflow = RightConstant([0., inflow_horizon], [demand, 0.], (0, float('inf')))
        network.add_commodity(0, 2, net_inflow, PredictorType.ZERO)
        times = evaluate_single_run(
            network,
            flow_id=None,
            focused_commodity=0,
            split=True,
            horizon=horizon,
            reroute_interval=reroute_interval,
            suppress_log=True,
            inflow_horizon=inflow_horizon,
            output_folder=None
        )
        for i, val in enumerate(times):
            avg_times[i].append(val)
        print(f"Calculated for demand={demand}. times={times}")
        demand += step_size
    print()
    print(avg_times)
    print()

    with open("./avg_times_sample.json", "w") as file:
        json.dump(avg_times, file)

    print("Successfully saved these travel times in ./avg_times_sample.json")
    print()
    sample_from_file_to_tikz()
    print("A tikz diagram was saved to ./avg_times_sample.tikz.")


def sample_from_file_to_tikz():
    configs = [
        {"label": "$\\hat q^{\\text{Z}}$", "color": "blue"},
        {"label": "$\\hat q^{\\text{C}}$", "color": "red"},
        {"label": "$\\hat q^{\\text{L}}$", "color": "{rgb,255:red,0; green,128; blue,0}"},
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
        legend pos=south east,
    ]
    """

    for c, values in enumerate(avg_times):
        tikz += "\n\\addplot[color=" + configs[c]["color"]
        if "dashed" in configs[c]:
            tikz += ", dashed"
        tikz += "]\n  coordinates { \n"

        for i, y in enumerate(values):
            x = i * 0.25
            tikz += f"({x}, {y})"

        tikz += "\n};\n"

    tikz += "\\end{axis}\n\\end{tikzpicture}"

    with open("./avg_times_sample.tikz", "w") as file:
        file.write(tikz)


if __name__ == '__main__':
    eval_sample()
