import datetime
from gc import callbacks
import os
import pickle
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from core.network import Network
from ml.QueueAndEdgeLoadsDataset import QueueAndEdgeLoadDataset


def train_full_net_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps, network: Network, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    dataset = QueueAndEdgeLoadDataset(queues_and_edge_loads_dir, past_timesteps, future_timesteps, network)

    X, Y = zip(*dataset)
    X, Y = np.array(X), np.array(Y)

    normalization = tf.keras.layers.Normalization()
    normalization.adapt(X)

    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        normalization,
        tf.keras.layers.Dense(units=2*X.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=X.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=X.shape[1] // 2),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=X.shape[1] // 4),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=X.shape[1] // 8),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=Y.shape[1] * 2),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=Y.shape[1]),
        tf.keras.layers.LeakyReLU()
    ])
    log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)


    model.summary()

    model.compile(loss='mean_absolute_error', optimizer=optimizer)

    model.fit(X_train, Y_train, epochs=1500, validation_data=(X_test, Y_test),  shuffle=True, callbacks=[tensorboard_callback, cp_callback])

    mlp: MLPRegressor = mlp.fit(X_train, Y_train)

    print(f"N_iter: {mlp.n_iter_}")
    print(f"Finished Fitting Training data.")
    print(f"Loss: {mlp.loss}")

    score = mlp.score(X_test, Y_test)
    Y_pred = mlp.predict(X_test)
    mse = mean_squared_error(Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred), squared=False)
    y_mean = np.mean(Y_test)
    print(f"Learned model with score {score}, RMSE={mse}, Y_mean={y_mean}")

    model = os.path.join(out_folder, "sklearn-full-net-model.pickle")
    with open(model, "wb") as file:
        pickle.dump(mlp, file)


if __name__ == '__main__':
    train_full_net_model()
