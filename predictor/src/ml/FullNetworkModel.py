from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.network import Network
from ml.DataLoader import QueueDataset


class FullNetworkModel(nn.Module):
    def __init__(self, in_features: int, num_edges: int, out_features: int):
        super(FullNetworkModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features * num_edges, in_features * out_features * num_edges),
            nn.ReLU(),
            nn.Linear(in_features * out_features * num_edges, in_features * out_features * num_edges),
            nn.ReLU(),
            nn.Linear(in_features * out_features * num_edges, out_features * num_edges)
        )

    def forward(self, input):
        return self.layers(input)


if __name__ == '__main__':
    torch_mode = 'cpu'
    network_path = '../../out/aaai-sioux-falls/network.pickle'
    network: Network = Network.from_file(network_path)
    past_timesteps, future_timesteps = 10, 10

    queue_folder_path = '../../out/aaai-sioux-falls/queues'
    queue_dataset = QueueDataset(queue_folder_path, past_timesteps, future_timesteps, network, False, torch_mode)
    num_edges = torch.count_nonzero(queue_dataset.test_mask).item()
    model = FullNetworkModel(past_timesteps, num_edges, future_timesteps).to(torch_mode)

    capacity = torch.from_numpy(network.capacity).float().to(torch_mode)[queue_dataset.test_mask]
    travel_time = torch.from_numpy(network.travel_time).float().to(torch_mode)[queue_dataset.test_mask]


    def collate(samples):
        input = torch.stack([sample[0] for sample in samples])
        label = torch.stack([sample[1] for sample in samples])
        return input, label


    batch_size = 64
    data_loader = DataLoader(queue_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fct = nn.L1Loss()
    model.train()

    epoch_losses = []
    for epoch in range(8000):
        epoch_loss = 0
        k = 0
        for k, (input, label) in enumerate(data_loader):
            optimizer.zero_grad()
            prediction = model(torch.flatten(input, start_dim=1))
            reshaped = torch.reshape(prediction, (input.shape[0], num_edges, future_timesteps))
            loss = loss_fct(reshaped, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            max_difference = torch.max(torch.abs(label - reshaped))
            rel_l1_error = torch.mean(torch.abs((label-reshaped) / torch.maximum(label, torch.full(label.shape, 1e-12))))

            print(
                f".. batch {k}, loss {loss:.4f}, rel_l1_err: {rel_l1_error:.4f}, max_diff {max_difference:.4f}, rolling avg: {epoch_loss / (k + 1):.4f}")

        epoch_loss /= (k + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        time = datetime.now().strftime("%H:%M:%S")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, f"../../out/{time}-{epoch}-{epoch_loss:.2f}.chk")
        epoch_losses.append(epoch_loss)
