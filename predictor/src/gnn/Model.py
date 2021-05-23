import dgl
import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch.utils.data import DataLoader

from gnn.DataLoader import QueueDataset
from importer.csv_importer import network_from_csv


class Model(nn.Module):
    def __init__(self, past_queue_steps: int, future_queue_steps: int, conv_layers: int):
        super(Model, self).__init__()
        assert conv_layers >= 2
        first_layer_dim = 2 + past_queue_steps
        middle_layer_dim = first_layer_dim * future_queue_steps * 60
        self.layers = nn.ModuleList([
            GraphConv(first_layer_dim, middle_layer_dim, activation=th.tanh)
        ])
        for i in range(1, conv_layers - 1):
            self.layers.append(GraphConv(middle_layer_dim, middle_layer_dim, activation=th.tanh))
        self.layers.append(GraphConv(middle_layer_dim, future_queue_steps))

    def forward(self, g, input):
        x = input
        for layer in self.layers:
            x = layer(g, x)
        return x


if __name__ == '__main__':
    network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
    network = network_from_csv(network_path)

    new_edges = [(e1.id, e2.id) for e1 in network.graph.edges for e2 in e1.node_to.outgoing_edges]
    u = th.tensor([e[0] for e in new_edges]).to('cuda')
    v = th.tensor([e[1] for e in new_edges]).to('cuda')
    graph = dgl.graph((u, v)).add_self_loop().to('cuda')

    past_timesteps, future_timesteps = 5, 5
    queue_folder_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/generated_queues/5,5/'
    queue_dataset = QueueDataset(queue_folder_path, past_timesteps, future_timesteps, network)

    capacity = th.from_numpy(network.capacity).float().to('cuda')
    travel_time = th.from_numpy(network.travel_time).float().to('cuda')

    conv_layers = 20
    model = Model(past_timesteps, future_timesteps, conv_layers).to('cuda')


    def collate(samples):
        th.stack([sample[0] for sample in samples])
        input = th.stack([th.column_stack([capacity, travel_time, sample[0]]) for sample in samples], dim=1)
        label = th.stack([sample[1] for sample in samples], dim=1)
        return input, label


    data_loader = DataLoader(queue_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    optimizer = th.optim.SGD(model.parameters(), lr=0.001)
    loss_fct = nn.L1Loss()

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        k = 0
        for k, (input, label) in enumerate(data_loader):
            print(f".. batch {k}, start computing")
            prediction = model(graph, input)
            loss = loss_fct(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f".. batch {k}, loss {loss:.4f}")
            epoch_loss += loss.detach().item()

        epoch_loss /= (k + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        th.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, f"./checkpoint.{epoch}.{epoch_loss:.2f}")
        epoch_losses.append(epoch_loss)
