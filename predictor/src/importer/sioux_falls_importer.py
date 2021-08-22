import pandas as pd

from core.network import Network


def import_sioux_falls(file_path: str, out_file_path: str):
    net = pd.read_csv(file_path, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    network = Network()
    for e in net:
        network.add_edge(e["init_node"], e["term_node"], e["free_flow_time"], e["capacity"])
    network.to_file(out_file_path)
