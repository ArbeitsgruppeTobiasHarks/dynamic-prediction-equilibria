import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from core.network import Network


def train_per_edge_model(
    network_path: str,
    expanded_queues_folder: str,
    out_folder: str,
    past_timesteps: int,
    future_timesteps: int,
):
    network: Network = Network.from_file(network_path)
    os.makedirs(out_folder, exist_ok=True)
    files = [
        file for file in os.listdir(expanded_queues_folder) if file.endswith(".csv.gz")
    ]
    for file in files:
        if file.startswith(".lock"):
            raise ValueError(f'Expanded Queue for file "{file}" has lock.')
        e_id = int(re.search(r"^edge-([0-9]+)\.csv\.gz$", file).groups()[0])
        model_path = os.path.join(out_folder, f"edge-{e_id}-model.pickle")
        lock_path = os.path.join(out_folder, f".lock.edge-{e_id}-model.pickle")

        if os.path.exists(model_path):
            print(f"Model for edge {e_id} already computed. Skipping...")
            continue
        elif os.path.exists(lock_path):
            print(f"Detected lock for edge {e_id}. Skipping...")
            continue

        with open(lock_path, "w") as f:
            f.write("")

        print(f"Computing model for edge {e_id}...")
        assert e_id in range(len(network.graph.edges))
        e = network.graph.edges[e_id]
        df = pd.read_csv(os.path.join(expanded_queues_folder, file))
        past_times = range(-past_timesteps + 1, 1)
        future_times = range(1, future_timesteps + 1)
        X_cols = (
            [f"{ie.id}[{t}]" for ie in e.node_from.incoming_edges for t in past_times]
            + [f"{oe.id}[{t}]" for oe in e.node_to.outgoing_edges for t in past_times]
            + [f"{e.id}[{t}]" for t in past_times]
        )
        Y_cols = [f"{e.id}[{t}]" for t in future_times]
        X, Y = df[X_cols].to_numpy(), df[Y_cols].to_numpy()

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        pipe = make_pipeline(MinMaxScaler(), Ridge())
        pipe = pipe.fit(X_train, Y_train)
        score = pipe.score(X_test, Y_test)
        Y_pred = pipe.predict(X_test)
        mse = mean_squared_error(
            Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred), squared=False
        )
        y_mean = np.mean(Y_test)
        print(
            f"Learned model for edge {e.id} with score {score}, RMSE={mse}, Y_mean={y_mean}"
        )

        with open(model_path, "wb") as file:
            pickle.dump(pipe, file)

        os.remove(lock_path)
