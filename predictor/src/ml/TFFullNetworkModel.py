import datetime
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from core.network import Network
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset
from utilities.file_lock import wait_for_locks, with_file_lock


def train_tf_full_net_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval: float, prediction_interval: float, horizon: float, network: Network, full_net_path: str, epochs: int = 10):
    test_mask = None
    def handle(_):
        os.makedirs(full_net_path, exist_ok=True)

        dataset = QueueAndEdgeLoadDataset(queues_and_edge_loads_dir, past_timesteps,
                                        future_timesteps, reroute_interval, prediction_interval, horizon, network)
        nonlocal test_mask
        test_mask = dataset.test_mask

        X, Y = zip(*dataset)
        X, Y = np.array(X), np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, shuffle=False)

        normalization = tf.keras.layers.Normalization()
        normalization.adapt(X_train)

        model: tf.keras.models.Sequential = tf.keras.models.Sequential([
            normalization,
            tf.keras.layers.Dense(units=X.shape[1],
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(units=X.shape[1],
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(units=X.shape[1],
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(units=X.shape[1] // 2,
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LeakyReLU()
        ])
        log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        optimizer = tf.keras.optimizers.Adam()

        checkpoint_path = "training/cp.ckpt"
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

        model.summary()

        model.compile(loss='mean_absolute_error', optimizer=optimizer)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3)

        model.fit(X_train, Y_train, epochs=epochs, validation_data=(
            X_test, Y_test),  shuffle=True, callbacks=[tensorboard_callback, cp_callback, early_stopping_callback])

        model.save(full_net_path)
        print(f"Finished Fitting Training data.")
    
    with_file_lock(full_net_path, handle)

    wait_for_locks(os.path.dirname(full_net_path))
    if test_mask is None:
        return QueueAndEdgeLoadDataset.load_mask(queues_and_edge_loads_dir)
    return test_mask
