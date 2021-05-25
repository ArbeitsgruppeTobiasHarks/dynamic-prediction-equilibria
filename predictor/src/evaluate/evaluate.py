from matplotlib import pyplot as plt

from core.constant_predictor import ConstantPredictor
from core.linear_predictor import LinearPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
from core.reg_linear_predictor import RegularizedLinearPredictor
from core.single_edge_distributor import SingleEdgeDistributor
from core.zero_predictor import ZeroPredictor
from test.sample_network import build_sample_network
from utilities.right_constant import RightConstantFunction


def evaluate_single_run(network: Network, split_commodity: int, horizon: float, reroute_interval: float):
    prediction_horizon = 0.05 * horizon

    predictors = [
        ConstantPredictor(network),
        ZeroPredictor(network),
        LinearPredictor(network, prediction_horizon),
        RegularizedLinearPredictor(network, prediction_horizon, delta=5.),
    ]

    commodity = network.commodities[split_commodity]
    network.commodities.remove(commodity)
    new_commodities = range(len(network.commodities), len(network.commodities) + len(predictors))
    for i in range(len(predictors)):
        network.commodities.append(Commodity(commodity.source, commodity.sink, commodity.demand / len(predictors), i))

    demand_per_comm = commodity.demand / len(predictors)
    distributor = SingleEdgeDistributor(network)
    flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)

    generator = flow_builder.build_flow()
    flow = next(generator)
    while flow.phi < horizon:
        flow = next(generator)

    travel_times = []

    for i in new_commodities:
        net_outflow: RightConstantFunction = sum(flow.outflow[e.id][i] for e in commodity.sink.incoming_edges)
        accum_net_outflow = net_outflow.integral()
        avg_travel_time = horizon / 2 - \
                          accum_net_outflow.integrate(0., horizon) / (horizon * demand_per_comm)
        travel_times.append(avg_travel_time)
    return travel_times


if __name__ == '__main__':

    y = [[], [], [], []]
    for demand in range(1, 20, 1):
        network = build_sample_network()
        network.add_commodity(0, 2, demand, 0)
        times = evaluate_single_run(network, 0, 100, 0.05)
        for i, time in enumerate(times):
            y[i].append(time)

    for i in range(len(y)):
        plt.plot(range(1, 20, 1), y[i], label=[
            "Constant Predictor",
            "Zero Predictor",
            "Linear Predictor",
            "Regularized Linear Predictor"
        ][i])
    plt.legend()
    plt.grid(which='both')
    plt.show()
