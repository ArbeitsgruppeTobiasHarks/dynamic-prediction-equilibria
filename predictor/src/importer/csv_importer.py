from numpy import genfromtxt

from core.network import Network


def network_from_csv(path: str) -> Network:
    np_data = genfromtxt(path, delimiter=' ', skip_header=1)
    # subsequent rows are structured as:
    # from_node to_node length_in_meters freeflow_speed_kph num_lanes
    # freeflow_speed_kph is -1 when the value is absent in the data.
    # We default to 20 for now.

    network = Network()
    for row in np_data:
        freeflow_speed = row[3] if row[3] > 0 else 20.
        travel_time = row[2] * freeflow_speed
        capacity = row[3] * freeflow_speed

        network.add_edge(row[0], row[1], travel_time, capacity)
    return network
