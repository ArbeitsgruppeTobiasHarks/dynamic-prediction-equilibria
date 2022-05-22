import os
import json
import json_fix

json_fix.fix_it()


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
