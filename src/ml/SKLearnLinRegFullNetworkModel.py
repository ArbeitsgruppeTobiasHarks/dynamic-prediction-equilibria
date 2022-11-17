import os
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline


from core.network import Network
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset


def train_full_net_model(queues_and_edge_loads_dir: str, past_timesteps: int, future_timesteps: int,
                         reroute_interval: float, prediction_interval: float, horizon: float, network: Network, model_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print("Full network model already exists. Skipping...")
        return QueueAndEdgeLoadDataset.load_mask(queues_and_edge_loads_dir)

    queue_dataset = QueueAndEdgeLoadDataset(
        queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval, prediction_interval, horizon, network)
    X, Y = zip(*queue_dataset)
    X, Y = np.array(X), np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    print(Y.shape)
    pipe = make_pipeline(
        preprocessing.MinMaxScaler(),
        MLPRegressor(max_iter=500, hidden_layer_sizes=(Y.shape[1]*8, Y.shape[1]*4, Y.shape[1]*2, Y.shape[1]), activation="relu"))
    pipe = pipe.fit(X_train, Y_train)
    score = pipe.score(X_test, Y_test)
    Y_pred = pipe.predict(X_test)
    mse = mean_squared_error(Y_test, np.maximum(
        np.zeros_like(Y_pred), Y_pred), squared=False)
    y_mean = np.mean(Y_test)
    print(f"Learned model with score {score}, RMSE={mse}, Y_mean={y_mean}")
    with open(model_path, "wb") as file:
        pickle.dump(pipe, file)
    return queue_dataset.test_mask


if __name__ == '__main__':
    train_full_net_model()
