import os
from core.nash_flow_builder import NashFlowBuilder
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from utilities.get_tn_path import get_tn_path
from visualization.to_json import to_visualization_json


def run_scenario(edges_tntp_path: str, nodes_tntp_path: str, od_pairs_file_path: str):
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    inflow_horizon = 12.
    add_od_pairs(network, od_pairs_file_path, inflow_horizon)
    network.commodities.sort(key=lambda c: (c.sink.id, next(iter(c.sources.keys())).id) )
    network.print_info()
    loader = NashFlowBuilder(network)
    flow, _ = loader.build_flow()
    to_visualization_json("./test.json", flow,  network,
                          {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'})


if __name__ == "__main__":
    def main():
        tn_path = get_tn_path()
        edges_tntp_path = os.path.join(
            tn_path, "SiouxFalls/SiouxFalls_net.tntp")
        nodes_tntp_path = os.path.expanduser(os.path.join(
            tn_path, "SiouxFalls/SiouxFalls_node.tntp"))
        od_pairs_csv_path = os.path.expanduser(os.path.join(
            tn_path, "SiouxFalls/CSV-data/SiouxFalls_od.csv"))
        run_scenario(edges_tntp_path, nodes_tntp_path, od_pairs_csv_path)

    main()
