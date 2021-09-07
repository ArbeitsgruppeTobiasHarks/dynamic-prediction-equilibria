import os
import pickle

import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from core.network import Network
from ml.DataLoader import QueueDataset


def train_model():
    torch_mode = 'cpu'
    out_folder = '../../out/aaai-sioux-falls'
    network_path = os.path.join(out_folder, 'network.pickle')
    network: Network = Network.from_file(network_path)
    past_timesteps, future_timesteps = 21, 20
    queue_folder_path = os.path.join(out_folder, 'queues')

    queue_dataset = QueueDataset(queue_folder_path, past_timesteps, future_timesteps, network, True, torch_mode)

    X, Y = zip(*queue_dataset)
    X = torch.row_stack([torch.flatten(x) for x in X])
    Y = torch.row_stack([torch.flatten(y) for y in Y])

    mlp = MLPRegressor(hidden_layer_sizes=(X.shape[1] + Y.shape[1],))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    mlp = mlp.fit(X_train, Y_train)
    model = os.path.join(out_folder, "sklearn-full-net-model.pickle")
    with open(model, "wb") as file:
        pickle.dump(mlp, file)


if __name__ == '__main__':
    train_model()
