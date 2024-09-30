import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WalkerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_categories):
        super().__init__()
        # Define the network layers
        emb1 = int(np.ceil(hidden_dim * .7))
        emb2 = hidden_dim - emb1
        self.embed = nn.Linear(input_dim, emb1)
        self.embed_bin = nn.Linear(num_categories, emb2)
        # self.embed_bin = nn.Embedding(num_categories, emb2)
        
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.is_active = nn.Linear(hidden_dim, 1)
        self.force = nn.Linear(hidden_dim, 1)
        self.choose_leg = nn.Linear(hidden_dim, num_categories)
        self.sig = nn.Sigmoid()

    def forward(self, x, switches):
        e = F.leaky_relu(self.embed(x))
        b = F.leaky_relu(self.embed_bin(switches.float()))
        # b = F.relu(self.embed_bin(switches)).mean(dim=1)
        x = torch.hstack((e, b))
        h = F.relu(self.hidden1(x))
        h = F.relu(self.hidden2(h))
        leg_probs = F.softmax(self.choose_leg(h), dim=1)
        active = self.sig(self.is_active(h))
        force = F.relu(self.force(h))
        return leg_probs, active, force
