""" Wrapper to use GNNs to learn graph observation representations. """

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Batch, DataLoader
from torch_geometric.nn import GCNConv


class ObsGraphRepresentation(gym.Wrapper):
    def __init__(self, env, n_epochs=50, n_samples=20000, 
                 validation_share=0.8, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        # Multi-step env would require to call step() here as well
        assert env.unwrapped.steps_per_episode == 1

        # Overwrite observation space: one observation per bus
        self.n_buses = len(env.net.bus)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.n_buses, 1), dtype=np.float32)
        
        self.n_epochs = n_epochs
        self.n_samples = n_samples

        train_loader, validation_batch = self.collect_data(env)
        self.train_model(train_loader, validation_batch)
        
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self.model.representation(info['graph_obs']).detach().numpy(), info
    
    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        obs = self.model.representation(info['next_graph_obs']).detach().numpy()
        return obs, reward, terminated, truncated, info

    def collect_data(self, env):
        data_list = []
        for i in range(self.n_samples):
            obs, info = env.reset()
            graph_obs = info['graph_obs']
            data_list.append(graph_obs)
        
        train_size = int(self.n_samples * 0.8)
        valid_size = self.n_samples - train_size
        train, validation = torch.utils.data.dataset.random_split(
            data_list, [train_size, valid_size])
        
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        return train_loader, Batch().from_data_list(validation)
    
    def train_model(self, train_loader, validation_batch):
        model = GraphAutoencoder(self.env.unwrapped.n_node_features)
        best_loss = np.inf
        for epoch in range(self.n_epochs):
            for batch in train_loader:
                train_loss = model.fit(batch)
            
            validation_loss = model.compute_loss(validation_batch)
            print(f'Epoch {epoch}: Loss {validation_loss}')

            # Always only store best model
            if validation_loss < best_loss:
                print('Store new model')
                best_loss = validation_loss
                self.model = model
        

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        # Reduce to one dimension per bus
        self.encoder = torch.nn.ModuleList([
            GCNConv(num_node_features, 4),
            GCNConv(4, 2),
            GCNConv(2, 1)
        ])

        # Bring back to original dimensions
        self.decoder = torch.nn.ModuleList([
            GCNConv(1, 2),
            GCNConv(2, 4),
            GCNConv(4, num_node_features)
        ])

        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, data):
        return self.encode(data)
    
    @torch.no_grad()
    def representation(self, data):
        return self.forward(data).squeeze()

    def encode(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.encoder[:-1]:
            x = layer(x, edge_index)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, training=self.training)

        return torch.nn.functional.tanh(self.encoder[-1](x, edge_index))
    
    def decode(self, x, edge_index):
        for layer in self.decoder[:-1]:
            x = layer(x, edge_index)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, training=self.training)

        return self.decoder[-1](x, edge_index)

    def fit(self, data):
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def compute_loss(self, data):
        x = self.encode(data)
        x = self.decode(x, data.edge_index)
        return torch.nn.functional.mse_loss(x, data.x)
