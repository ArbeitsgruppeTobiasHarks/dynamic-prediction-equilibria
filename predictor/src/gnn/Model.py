from datetime import datetime

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv
from torch.utils.data import DataLoader

from gnn.DataLoader import QueueDataset
from importer.csv_importer import network_from_csv


class Model(nn.Module):
    def __init__(self, past_queue_steps: int, future_queue_steps: int, conv_layers: int):
        super(Model, self).__init__()
        assert conv_layers >= 2
        first_layer_dim = 2 + past_queue_steps
        middle_layer_dim = 1000
        self.layers = nn.ModuleList([
            GraphConv(first_layer_dim, middle_layer_dim, norm='none', activation=torch.relu)
        ])
        for i in range(1, conv_layers - 1):
            self.layers.append(GraphConv(middle_layer_dim, middle_layer_dim, norm='none', activation=torch.relu))
        self.layers.append(GraphConv(middle_layer_dim, future_queue_steps, norm='none'))

    def forward(self, g, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(g.reverse() if i % 4 == 3 else g, x)
        return x


class DenseLayerModel(nn.Module):
    def __init__(self, in_features: int, num_edges: int, out_features: int):
        super(DenseLayerModel, self).__init__()
        self.first = nn.Linear(in_features * num_edges, out_features * num_edges)

    def forward(self, input):
        return self.first(input)


if __name__ == '__main__':
    network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
    network = network_from_csv(network_path)

    new_edges = [(e1.id, e2.id) for e1 in network.graph.edges for e2 in e1.node_to.outgoing_edges]
    u = torch.tensor([e[0] for e in new_edges]).to('cuda')
    v = torch.tensor([e[1] for e in new_edges]).to('cuda')
    graph = dgl.graph((u, v)).add_self_loop().to('cuda')

    past_timesteps, future_timesteps = 5, 1
    conv_layers = 6
    queue_folder_path = '../../out/generated_queues/5,5/'
    queue_dataset = QueueDataset(queue_folder_path, past_timesteps, future_timesteps, network, False)

    mask = queue_dataset.test_mask.cpu().numpy()
    np.savetxt('/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/generated_queues/5,5/mask.txt', mask,
               fmt="%d")

    # subgraph_nodes = torch.nonzero(queue_dataset.test_mask, as_tuple=True)[0]
    # graph = dgl.node_subgraph(graph, subgraph_nodes)
    num_edges = torch.count_nonzero(queue_dataset.test_mask).item()

    # model = Model(past_timesteps, future_timesteps, conv_layers).to('cuda')
    model = DenseLayerModel(past_timesteps, num_edges, future_timesteps).to('cuda')

    capacity = torch.from_numpy(network.capacity).float().to('cuda')[queue_dataset.test_mask]
    travel_time = torch.from_numpy(network.travel_time).float().to('cuda')[queue_dataset.test_mask]


    def collate(samples):
        input = torch.stack([sample[0] for sample in samples], dim=1)
        label = torch.stack([sample[1] for sample in samples], dim=1)
        return input, label


    data_loader = DataLoader(queue_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fct = nn.MSELoss()
    model.train()

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        k = 0
        for k, (input, label) in enumerate(data_loader):
            optimizer.zero_grad()
            prediction = model(torch.flatten(input))
            reshaped = torch.reshape(prediction, (num_edges, future_timesteps))
            loss = loss_fct(reshaped, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            max_difference = torch.max(torch.abs(label - reshaped))

            print(
                f".. batch {k}, loss {loss:.4f}, max_diff {max_difference:.4f}, rolling avg: {epoch_loss / (k + 1):.4f}")

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
