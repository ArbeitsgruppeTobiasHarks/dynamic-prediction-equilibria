import gzip
import os
from typing import Collection, Dict

from core.dynamic_flow import DynamicFlow, FlowRatesCollection
from core.network import Network
from utilities.json_encoder import JSONEncoder
from utilities.right_constant import RightConstant


def merge_commodities(
    flow: DynamicFlow, network: Network, commodities: Collection[int]
) -> DynamicFlow:
    if len(commodities) == 0:
        return flow
    new_comm_id = min(commodities)
    merged_flow = DynamicFlow(network)
    merged_flow.queues = flow.queues

    def merge_collection(old: FlowRatesCollection) -> FlowRatesCollection:
        new_d: Dict[int, RightConstant] = {}
        for i, right_constant in old._functions_dict.items():
            if i not in commodities:
                new_d[i] = right_constant
            elif new_comm_id not in new_d:
                new_d[new_comm_id] = right_constant
            else:
                new_d[new_comm_id] += right_constant
        if new_comm_id in new_d:
            new_d[new_comm_id] = new_d[new_comm_id].simplify()
        return FlowRatesCollection(new_d)

    merged_flow.inflow = [merge_collection(v) for v in flow.inflow]

    merged_flow.outflow = [merge_collection(v) for v in flow.outflow]
    return merged_flow


def to_visualization_json(
    path: str, flow: DynamicFlow, network: Network, colors: Dict[int, str]
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="UTF-8") as file:
        JSONEncoder().dump(
            {
                "network": {
                    "nodes": [
                        {
                            "id": id,
                            "x": network.graph.positions[id][0],
                            "y": network.graph.positions[id][1],
                        }
                        for (id, v) in network.graph.nodes.items()
                    ],
                    "edges": [
                        {
                            "id": id,
                            "from": e.node_from.id,
                            "to": e.node_to.id,
                            "capacity": network.capacity[id],
                            "transitTime": network.travel_time[id],
                        }
                        for (id, e) in enumerate(network.graph.edges)
                    ],
                    "commodities": [
                        {"id": id, "color": colors[id]}
                        for (id, comm) in enumerate(network.commodities)
                    ],
                },
                "flow": {
                    "inflow": [col._functions_dict for col in flow.inflow],
                    "outflow": [col._functions_dict for col in flow.outflow],
                    "queues": flow.queues,
                },
            },
            file,
        )
