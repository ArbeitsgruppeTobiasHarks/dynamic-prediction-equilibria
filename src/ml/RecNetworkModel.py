import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.network import Network
from ml.QueueDataset import QueueDataset


class RNNModel(nn.Module):
    def __init__(self, num_edges: int):
        super(RNNModel, self).__init__()
        hidden_layer_size = num_edges * 10
        self.rnn = nn.RNN(
            num_edges, hidden_layer_size, nonlinearity="relu", batch_first=True
        )
        self.hidden_state = None
        self.linear = nn.Linear(hidden_layer_size, num_edges)
        self.relu = nn.ReLU()

    def forward(self, input):
        out, self.hidden_state = self.rnn(input, self.hidden_state)
        return self.relu(self.linear(out[:, -1, :]))


def train_model():
    torch_mode = "cpu"
    out_folder = "../../out/aaai-sioux-falls"
    network_path = os.path.join(out_folder, "network.pickle")
    network: Network = Network.from_file(network_path)
    past_timesteps, future_timesteps = 20, 1
    queue_folder_path = os.path.join(out_folder, "queues")

    queue_dataset = QueueDataset(
        queue_folder_path, past_timesteps, future_timesteps, network, False, torch_mode
    )
    num_edges = torch.count_nonzero(queue_dataset.test_mask).item()
    model = RNNModel(num_edges).to(torch_mode)

    checkpoints_dir = os.path.join(out_folder, "rnn-model")
    os.makedirs(checkpoints_dir, exist_ok=True)

    def collate(samples):
        input = torch.stack([torch.swapaxes(sample[0], 0, 1) for sample in samples])
        label = torch.stack([torch.swapaxes(sample[1], 0, 1) for sample in samples])
        return input, label

    batch_size = 64
    data_loader = DataLoader(
        queue_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fct = nn.MSELoss()
    model.train()

    epoch_losses = []
    for epoch in range(8000):
        epoch_loss = 0
        k = 0
        for k, (input, label) in enumerate(data_loader):
            assert future_timesteps == 1
            label = torch.reshape(label, (label.shape[0], num_edges))
            prediction = model(input)
            model.hidden_state = None
            optimizer.zero_grad()
            loss = loss_fct(prediction, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            max_difference = torch.max(torch.abs(label - prediction))
            rel_l1_error = torch.mean(
                torch.abs(
                    (label - prediction)
                    / torch.maximum(label, torch.full(label.shape, 1e-12))
                )
            )

            print(
                f".. batch {k}, loss {loss:.4f}, rel_l1_err: {rel_l1_error:.4f}, max_diff {max_difference:.4f}, rolling avg: {epoch_loss / (k + 1):.4f}"
            )

        epoch_loss /= k + 1
        print("Epoch {}, loss {:.4f}".format(epoch, epoch_loss))
        time = datetime.now().strftime("%H:%M:%S")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            os.path.join(checkpoints_dir, f"{time}-{epoch}-{epoch_loss:.2f}.chk"),
        )
        epoch_losses.append(epoch_loss)


if __name__ == "__main__":
    train_model()
