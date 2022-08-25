import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from core.network import Network
from core.predictors.sk_neighborhood_predictor import SKNeighborhoodPredictor
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset
from ml.neighboring_edges import get_neighboring_edges_mask_undirected
from utilities.file_lock import wait_for_locks, with_file_lock


def train_sk_neighborhood_model(
        queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval: float,
        prediction_interval: float, horizon: float, network: Network, models_path: str,
        max_distance: int):
    os.makedirs(models_path, exist_ok=True)
    dataset = None

    input_mask, output_mask = None, None

    if not QueueAndEdgeLoadDataset.mask_exists(queues_and_edge_loads_dir):
        dataset = QueueAndEdgeLoadDataset(queues_and_edge_loads_dir, past_timesteps,
                                          future_timesteps, reroute_interval, prediction_interval, horizon, network)
        input_mask, output_mask = dataset.input_mask, dataset.output_mask
    else:
        input_mask, output_mask = QueueAndEdgeLoadDataset.load_mask(
            queues_and_edge_loads_dir)

    for edge in network.graph.edges:
        if not output_mask[edge.id]:
            continue
        model_path = os.path.join(models_path, str(edge.id) + ".pickle")

        def handle(_):
            nonlocal dataset, input_mask, output_mask

            if dataset is None:
                dataset = QueueAndEdgeLoadDataset(queues_and_edge_loads_dir, past_timesteps,
                                                  future_timesteps, reroute_interval, prediction_interval, horizon, network)
            input_mask, output_mask = dataset.input_mask, dataset.output_mask
            edge_input_mask = input_mask * \
                get_neighboring_edges_mask_undirected(
                    edge, network, max_distance)
            edge_output_mask = np.array(
                [e == edge for e in network.graph.edges])

            dataset.use_additional_input_mask(edge_input_mask)
            dataset.use_additional_output_mask(edge_output_mask)

            X, Y = zip(*dataset)
            X, Y = np.array(X), np.array(Y)

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.1, shuffle=False)

            pipe = make_pipeline(MinMaxScaler(), Ridge())
            pipe = pipe.fit(X_train, Y_train)
            score = pipe.score(X_test, Y_test)
            Y_pred = pipe.predict(X_test)
            mae = mean_absolute_error(
                Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred))
            # mse = mean_squared_error(Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred), squared=False)
            # y_mean = np.mean(Y_test)

            with open(model_path, "wb") as file:
                pickle.dump(pipe, file)

            print(
                f"Learned model for edge {edge.id} with score {score}, MAE={mae}")

        with_file_lock(model_path, handle)

    wait_for_locks(os.path.dirname(models_path))

    def build_sk_neighborhood_predictor(network: Network) -> SKNeighborhoodPredictor:
        return SKNeighborhoodPredictor.from_models(
            network, models_path, input_mask, output_mask, past_timesteps, future_timesteps, prediction_interval, max_distance)
    return build_sk_neighborhood_predictor
