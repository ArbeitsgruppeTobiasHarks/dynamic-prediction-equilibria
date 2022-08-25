import os
import pickle
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from core.network import Network
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset
from utilities.file_lock import wait_for_locks, with_file_lock


def train_sk_full_net_model(
        queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval: float,
        prediction_interval: float, horizon: float, network: Network, full_net_path: str):
    input_mask = None
    output_mask = None

    def handle(_):
        dataset = QueueAndEdgeLoadDataset(queues_and_edge_loads_dir, past_timesteps,
                                          future_timesteps, reroute_interval, prediction_interval, horizon, network)
        nonlocal input_mask, output_mask
        input_mask = dataset.input_mask
        output_mask = dataset.output_mask

        X, Y = zip(*dataset)
        X, Y = np.array(X), np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

        ridge = Ridge()
        pipe = make_pipeline(MinMaxScaler(), ridge)
        pipe = pipe.fit(X_train, Y_train)
        score = pipe.score(X_test, Y_test)
        Y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred))
        # mse = mean_squared_error(Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred), squared=False)
        # y_mean = np.mean(Y_test)

        with open(full_net_path, "wb") as file:
            pickle.dump(pipe, file)
        
        print(f"Learned model with score {score}, MAE={mae}")

        highest = sorted(enumerate(np.mean(np.absolute(ridge.coef_), axis=0)), key=lambda x: x[1], reverse=True) 
        print(f"Highest (mean absolute) 10 Ridge coefficient indices: {highest[:10]}")

    with_file_lock(full_net_path, handle)

    wait_for_locks(os.path.dirname(full_net_path))
    if input_mask is None or output_mask is None:
        return QueueAndEdgeLoadDataset.load_mask(queues_and_edge_loads_dir)
    return input_mask, output_mask
