import os
import json
from typing import Dict, Set
import json_fix

from core.dynamic_flow import DynamicFlow
from core.network import Network
from utilities.right_constant import RightConstant

json_fix.fix_it()


def merge_commodities(flow: DynamicFlow, network: Network, commodities: Set[int]) -> DynamicFlow:
    if len(commodities) == 0:
        return flow, network
    new_comm_id = min(commodities)
    merged_flow = DynamicFlow(network)
    merged_flow.queues = flow.queues

    def merge_dict(d: Dict[int, RightConstant]) -> Dict[int, RightConstant]:
        new_d: Dict[int, RightConstant] = {}
        for i, right_constant in d.items():
            if i not in commodities:
                new_d[i] = right_constant
            elif new_comm_id not in new_d:
                new_d[new_comm_id] = right_constant
            else:
                new_d[new_comm_id] += right_constant
        if new_comm_id in new_d:
            new_d[new_comm_id] = new_d[new_comm_id].simplify()
        return new_d

    merged_flow.inflow = [
        merge_dict(v)
        for v in flow.inflow
    ]

    merged_flow.outflow = [
        merge_dict(v)
        for v in flow.outflow
    ]
    return merged_flow


def to_visualization_json(path, flow, network, colors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump({
            "network": {
                "nodes": [
                    {"id": id, "x": network.graph.positions[id]
                        [0], "y": network.graph.positions[id][1]}
                    for (id, v) in network.graph.nodes.items()
                ],
                "edges": [
                    {
                        "id": id,
                        "from": e.node_from.id,
                        "to": e.node_to.id,
                        "capacity": network.capacity[id],
                        "transitTime": network.travel_time[id]
                    }
                    for (id, e) in enumerate(network.graph.edges)
                ],
                "commodities": [
                    {"id": id, "color": colors[id]}
                    for (id, comm) in enumerate(network.commodities)
                ]
            },
            "flow": {
                "inflow": flow.inflow,
                "outflow": flow.outflow,
                "queues": flow.queues
            }}, file)
