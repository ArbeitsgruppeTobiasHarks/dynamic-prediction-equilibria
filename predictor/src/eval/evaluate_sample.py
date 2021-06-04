import json

from eval.evaluate import evaluate_single_run
from eval.sample_network import build_sample_network


def eval_sample():
    max_demand = 30.
    demand = 0.
    step_size = 0.25
    avg_times = [[], [], [], [], []]
    while demand < max_demand:
        network = build_sample_network()
        network.add_commodity(0, 2, demand, 0)
        times = evaluate_single_run(network,
                                    flow_id=None,
                                    focused_commodity=0,
                                    split=True, horizon=100, reroute_interval=0.25,
                                    suppress_log=True,
                                    output_folder=None)
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


def sample_from_file_to_tikz():
    with open("./avg_times_sample.json", "r") as file:
        avg_times = json.load(file)
    for values in avg_times:
        tikz = ""
        for i, y in enumerate(values):
            x = i * 0.25
            tikz += f"({x}, {y})"

        print(tikz)
