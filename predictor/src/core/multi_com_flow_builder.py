from __future__ import annotations

from typing import Generator, Optional, Dict, List, Set

import numpy as np
from functools import reduce

from core.bellman_ford import bellman_ford
from core.constant_predictor import ConstantPredictor
from core.dijkstra import dijkstra
from core.distributor import Distributor
from core.graph import Node
from core.linear_regression_predictor import LinearRegressionPredictor
from core.machine_precision import eps
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from utilities.interpolate import LinearlyInterpolatedFunction
from utilities.queues import PriorityQueue


class MultiComFlowBuilder:
    network: Network
    predictors: List[Predictor]
    distributor: Distributor
    reroute_interval: Optional[float]

    def __init__(self, network: Network,
                 predictors: List[Predictor],
                 distributor: Distributor,
                 reroute_interval: Optional[float]  # None means rerouting every time some outflow changes
                 ):
        self.network = network
        self.predictors = predictors
        self.reroute_interval = reroute_interval
        self.distributor = distributor

    def build_flow(self) -> Generator[MultiComPartialDynamicFlow, None, None]:
        flow = MultiComPartialDynamicFlow(self.network)
        n = len(self.network.commodities)
        m = len(self.network.graph.edges)
        travel_time = self.network.travel_time
        capacity = self.network.capacity

        # Preprocessing...
        # For each commodity find the nodes that reach the sink
        reaching_nodes = [
            self.network.graph.get_nodes_reaching(commodity.sink) for commodity in self.network.commodities
        ]
        reachable_nodes = [
            self.network.graph.get_reachable_nodes(commodity.source) for commodity in self.network.commodities
        ]
        important_nodes = [
            reaching_nodes[i].intersection(reachable_nodes[i]) for i in range(len(self.network.commodities))
        ]
        assert all(c.source in reaching_nodes[i] for i, c in enumerate(self.network.commodities))

        next_reroute_time = flow.phi
        costs = []
        labels: Dict[int, Dict[Node, LinearlyInterpolatedFunction]] = {}
        const_labels = {}
        const_costs = {}
        handle_nodes = set(self.network.graph.nodes.values())
        queues = None

        yield flow
        while flow.phi < float('inf'):
            if self.reroute_interval is None or flow.phi >= next_reroute_time - eps:
                # PREDICT NEW QUEUES
                predictions = [predictor.predict_from_fcts(flow.queues, flow.phi) for predictor in self.predictors]
                pred_queues_list = [np.asarray(prediction.queues) for prediction in predictions]
                pred_costs = [[travel_time[e] + pred_queues[:, e] / capacity[e] for e in range(m)]
                              for pred_queues in pred_queues_list]
                costs = [
                    [
                        LinearlyInterpolatedFunction(predictions[k].times, pred_costs[k][e], (flow.phi, float('inf')))
                        for e in range(m)
                    ]
                    for k in range(len(self.predictors))]

                const_costs = None
                # CALCULATE NEW SHORTEST PATHS
                for i, commodity in enumerate(self.network.commodities):
                    if isinstance(self.predictors[commodity.predictor], ConstantPredictor):
                        # Handle constant predictor separately for better performance
                        if const_costs is None:
                            const_costs = travel_time + pred_queues_list[commodity.predictor][0, :] / capacity
                        const_labels[i] = dijkstra(commodity.sink, const_costs)
                        if self.distributor.supports_const():
                            continue

                        labels[i] = {
                            v: LinearlyInterpolatedFunction([flow.phi, flow.phi + 1], [label, label],
                                                            (flow.phi, float('inf')))
                            for v, label in const_labels[i].items()
                        }
                    elif isinstance(self.predictors[commodity.predictor], LinearRegressionPredictor):
                        # We have an own distributor which cares about finding shortest paths.
                        pass
                    else:
                        labels[i] = bellman_ford(
                            commodity.sink, costs[commodity.predictor], important_nodes[i], flow.phi
                        )
                next_reroute_time += self.reroute_interval
                handle_nodes = set(self.network.graph.nodes.values())

            # DETERMINE OUTFLOW SPLIT
            if self.distributor.needs_queues():
                queues = np.asarray([queue(flow.phi) for queue in flow.queues])

            inflow_per_comm: List[Dict[int, float]] = []
            for i, commodity in enumerate(self.network.commodities):
                node_inflow: Dict[Node, float] = {
                    v: sum(flow.outflow[e.id][i](flow.phi) for e in v.incoming_edges)
                    for v in important_nodes[i].intersection(handle_nodes)
                }
                if commodity.source in handle_nodes:
                    node_inflow[commodity.source] += commodity.demand

                if isinstance(self.predictors[commodity.predictor], ConstantPredictor) and \
                        self.distributor.supports_const():
                    new_inflow_i = self.distributor.distribute_const(
                        flow.phi, node_inflow, commodity.sink, queues, const_labels[i], const_costs
                    )
                    inflow_per_comm.append(new_inflow_i)
                elif isinstance(self.predictors[commodity.predictor], LinearRegressionPredictor):
                    new_inflow_i = self.distribute_lin_reg(next_reroute_time - self.reroute_interval, node_inflow,
                                                           commodity.sink, important_nodes[i],
                                                           costs[commodity.predictor])
                    inflow_per_comm.append(new_inflow_i)
                else:
                    new_inflow_i = self.distributor.distribute(flow.phi, node_inflow, commodity.sink, queues, labels[i],
                                                               costs[commodity.predictor])
                    inflow_per_comm.append(new_inflow_i)
            all_keys = reduce(lambda acc, item: acc.union(item.keys()), inflow_per_comm, set())
            new_inflow = {
                e: np.asarray([
                    inflow_per_comm[i][e]
                    if e in inflow_per_comm[i].keys() else
                    flow.inflow[e][i].values[-1]
                    for i in range(n)
                ]) for e in all_keys
            }

            max_ext_length = next_reroute_time - flow.phi
            changed_edges = flow.extend(new_inflow, max_ext_length)

            yield flow

            handle_nodes = set()
            for e in changed_edges:
                handle_nodes.add(self.network.graph.edges[e].node_to)

    def distribute_lin_reg(self, phi: float, node_inflow: Dict[Node, float], sink: Node, interesting_nodes: Set[Node],
                           costs: List[LinearlyInterpolatedFunction]) -> Dict[int, float]:
        new_inflow = {}
        for s in node_inflow.keys():
            if node_inflow[s] < eps:
                for e in s.outgoing_edges:
                    new_inflow[e.id] = 0.
                continue

            # Do Time-Dependent dijkstra from s to t to find active outgoing edges of s

            arrival_times: Dict[Node, float] = {s: phi}
            queue: PriorityQueue[Node] = PriorityQueue([(s, phi)])
            realised_cost = {}
            stop_after = float('inf')
            while len(queue) > 0:
                arrival_time = queue.min_key()
                v = queue.pop()
                if v == sink:
                    stop_after = arrival_time
                if arrival_time > stop_after:
                    break

                for e in v.outgoing_edges:
                    w = e.node_to
                    if w not in interesting_nodes:
                        continue
                    realised_cost[e] = costs[e.id](arrival_time)
                    relaxation = arrival_times[v] + realised_cost[e]
                    if w not in arrival_times.keys():
                        arrival_times[w] = relaxation
                        queue.push(w, relaxation)
                    elif relaxation < arrival_times[w]:
                        arrival_times[w] = relaxation
                        queue.decrease_key(w, relaxation)

            # Dijkstra done. Now searching all active edges leading to t.
            active_edges = []
            touched_nodes = {sink}
            queue: List[Node] = [sink]
            while queue:
                w = queue.pop()
                for e in w.incoming_edges:
                    if e not in realised_cost.keys():
                        continue
                    v: Node = e.node_from
                    if arrival_times[v] + realised_cost[e] <= arrival_times[w] + eps:
                        if v == s:
                            active_edges.append(e)
                        if v not in touched_nodes:
                            touched_nodes.add(v)
                            queue.append(v)

            assert len(active_edges) > 0

            distribution = node_inflow[s] / len(active_edges)
            for e in active_edges:
                new_inflow[e.id] = distribution

        return new_inflow
