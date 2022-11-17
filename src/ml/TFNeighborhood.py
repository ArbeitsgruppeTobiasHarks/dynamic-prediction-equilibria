import datetime
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from core.network import Network
from core.predictors.tf_neighborhood_predictor import TFNeighborhoodPredictor
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset
from ml.neighboring_edges import get_neighboring_edges_mask_undirected
from utilities.file_lock import wait_for_locks, with_file_lock

def train_tf_neighborhood_model(
        queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval: float,
        prediction_interval: float, horizon: float, network: Network, models_path: str,
        max_distance: int, epochs: int = 10):
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
        if output_mask[edge.id] == 0:
            continue
        model_dir = os.path.join(models_path, str(edge.id))

        def handle(_):
            os.makedirs(model_dir, exist_ok=True)
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

            normalization = tf.keras.layers.Normalization()
            normalization.adapt(X_train)

            model: tf.keras.models.Sequential = tf.keras.models.Sequential([
                normalization,
                tf.keras.layers.Dense(units=X.shape[1],
                                      kernel_regularizer=tf.keras.regularizers.l2(
                                          0.001),
                                      bias_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(units=X.shape[1],
                                      kernel_regularizer=tf.keras.regularizers.l2(
                                          0.001),
                                      bias_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(units=X.shape[1],
                                      kernel_regularizer=tf.keras.regularizers.l2(
                                          0.001),
                                      bias_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(units=Y.shape[1],
                                      kernel_regularizer=tf.keras.regularizers.l2(
                                          0.001),
                                      bias_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.LeakyReLU()
            ])
            log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            optimizer = tf.keras.optimizers.Adam()

            checkpoint_path = "training/cp.ckpt"
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path)

            model.summary()

            model.compile(loss='mean_absolute_error', optimizer=optimizer)

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=3)

            model.fit(X_train, Y_train, epochs=epochs, validation_data=(
                X_test, Y_test),  shuffle=True, callbacks=[tensorboard_callback, cp_callback, early_stopping_callback])

            model.save(model_dir)
            print(f"Finished Fitting Training data for edge {edge.id}.")

        with_file_lock(model_dir, handle)

    wait_for_locks(os.path.dirname(models_path))

    def build_tf_neighborhood_predictor(network: Network) -> TFNeighborhoodPredictor:
        return TFNeighborhoodPredictor.from_models(network, models_path, input_mask, output_mask, past_timesteps, future_timesteps, prediction_interval, max_distance)

    return build_tf_neighborhood_predictor
