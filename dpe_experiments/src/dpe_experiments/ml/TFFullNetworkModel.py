import datetime
import os

import keras
import numpy as np
from dynflows.core.network import Network
from dynflows.utilities.file_lock import wait_for_locks, with_file_lock
from sklearn.model_selection import train_test_split

from dpe_experiments.ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset
from dpe_experiments.predictors.tf_full_net_predictor import TFFullNetPredictor


def train_tf_full_net_model(
    queues_and_edge_loads_dir,
    past_timesteps,
    future_timesteps,
    reroute_interval: float,
    prediction_interval: float,
    horizon: float,
    network: Network,
    full_net_path: str,
    epochs: int = 10,
):
    input_mask = None
    output_mask = None

    def handle(_):
        os.makedirs(full_net_path, exist_ok=True)

        dataset = QueueAndEdgeLoadDataset(
            queues_and_edge_loads_dir,
            past_timesteps,
            future_timesteps,
            reroute_interval,
            prediction_interval,
            horizon,
            network,
        )
        nonlocal input_mask, output_mask
        input_mask = dataset.input_mask
        output_mask = dataset.output_mask

        X, Y = zip(*dataset)
        X, Y = np.array(X), np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, shuffle=False
        )

        normalization = keras.layers.Normalization()
        normalization.adapt(X_train)

        model: keras.models.Sequential = keras.models.Sequential(
            [
                normalization,
                keras.layers.Dense(
                    units=X.shape[1],
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    bias_regularizer=keras.regularizers.l2(0.001),
                ),
                keras.layers.LeakyReLU(),
                keras.layers.Dense(
                    units=X.shape[1],
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    bias_regularizer=keras.regularizers.l2(0.001),
                ),
                keras.layers.LeakyReLU(),
                keras.layers.Dense(
                    units=X.shape[1],
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    bias_regularizer=keras.regularizers.l2(0.001),
                ),
                keras.layers.LeakyReLU(),
                keras.layers.Dense(
                    units=Y.shape[1],
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    bias_regularizer=keras.regularizers.l2(0.001),
                ),
                keras.layers.LeakyReLU(),
            ]
        )
        log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        optimizer = keras.optimizers.Adam()

        checkpoint_path = os.path.join(full_net_path, "cp.keras")
        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

        model.summary()

        model.compile(loss="mean_absolute_error", optimizer=optimizer)

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="loss", patience=3
        )

        model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            shuffle=True,
            callbacks=[tensorboard_callback, cp_callback, early_stopping_callback],
        )

        model.save(os.path.join(full_net_path, "model.keras"))
        print(f"Finished Fitting Training data.")

    with_file_lock(
        full_net_path,
        handle,
        expect_exists=[os.path.join(full_net_path, "model.keras")],
    )

    wait_for_locks(os.path.dirname(full_net_path))
    if input_mask is None or output_mask is None:
        input_mask, output_mask = QueueAndEdgeLoadDataset.load_mask(
            queues_and_edge_loads_dir
        )

    def build_tf_full_net_predictor(network: Network) -> TFFullNetPredictor:
        return TFFullNetPredictor.from_model(
            network,
            os.path.join(full_net_path, "model.keras"),
            input_mask,
            output_mask,
            past_timesteps,
            future_timesteps,
            prediction_interval,
        )

    return build_tf_full_net_predictor
