import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ThrowerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Define the network layers
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        # self.force = nn.Linear(hidden_dim, 1)
        
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.direction = nn.Linear(hidden_dim, output_dim)
        
        self.tanh = nn.Tanh()
        self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.l_relu(self.embed(x))
        h1 = self.l_relu(self.hidden1(x))
        h2 = self.l_relu(self.hidden2(h1))
        # f = self.l_relu(self.force(h1))
        d = self.direction(h2)
        return d
