import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.network import Network


def train_per_edge_model(network_path: str, expanded_queues_folder: str, out_folder: str, past_timesteps: int,
                         future_timesteps: int):
    network: Network = Network.from_file(network_path)
    os.makedirs(out_folder, exist_ok=True)
    files = [file for file in os.listdir(expanded_queues_folder) if file.endswith(".csv.gz")]
    for file in files:
        if file.startswith(".lock"):
            raise ValueError("Expanded Queue has lock.")
        e_id = int(re.search(r"^edge-([0-9]+)\.csv\.gz$", file).groups()[0])
        assert e_id in range(len(network.graph.edges))
        e = network.graph.edges[e_id]
        df = pd.read_csv(os.path.join(expanded_queues_folder, file))
        past_times = range(-past_timesteps + 1, 1)
        future_times = range(1, future_timesteps + 1)
        X_cols = [f"{ie.id}[{t}]" for ie in e.node_from.incoming_edges for t in past_times] + \
                 [f"{oe.id}[{t}]" for oe in e.node_to.outgoing_edges for t in past_times] + \
                 [f"{e.id}[{t}]" for t in past_times]
        Y_cols = [f"{e.id}[{t}]" for t in future_times]
        X, Y = df[X_cols].to_numpy(), df[Y_cols].to_numpy()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        pipe = make_pipeline(StandardScaler(), MLPRegressor())
        pipe = pipe.fit(X_train, Y_train)
        score = pipe.score(X_test, Y_test)
        print(f"Learned model for edge {e.id} with score {score}.")
        model = os.path.join(out_folder, f"edge-{e.id}-model.pickle")
        with open(model, "wb") as file:
            pickle.dump(pipe, file)
