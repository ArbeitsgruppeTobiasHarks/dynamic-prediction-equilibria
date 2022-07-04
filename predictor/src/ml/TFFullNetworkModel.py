import datetime
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from core.network import Network
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset


def train_tf_full_net_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval: float, prediction_interval: float, horizon: float, network: Network, full_net_path: str):
    epochs = 10

    if os.path.exists(full_net_path):
        print("Full network model already exists. Skipping...")
        return QueueAndEdgeLoadDataset.load_mask(queues_and_edge_loads_dir)
    os.makedirs(full_net_path, exist_ok=True)

    dataset = QueueAndEdgeLoadDataset(queues_and_edge_loads_dir, past_timesteps,
                                      future_timesteps, reroute_interval, prediction_interval, horizon, network)

    X, Y = zip(*dataset)
    X, Y = np.array(X), np.array(Y)

    normalization = tf.keras.layers.Normalization()
    normalization.adapt(X)

    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        normalization,
        tf.keras.layers.Dense(units=4*X.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=2*X.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=X.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=X.shape[1] // 2),
        tf.keras.layers.LeakyReLU()
    ])
    log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

    model.summary()

    model.compile(loss='mean_absolute_error', optimizer=optimizer)

    model.fit(X_train, Y_train, epochs=epochs, validation_data=(
        X_test, Y_test),  shuffle=True, callbacks=[tensorboard_callback, cp_callback])

    model.save(full_net_path)
    print(f"Finished Fitting Training data.")
    return dataset.test_mask


if __name__ == '__main__':
    train_tf_full_net_model()
