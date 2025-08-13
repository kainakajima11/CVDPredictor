import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim):
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, 64)
        self.fc2: nn.Linear = nn.Linear(64, 32)
        self.fc3: nn.Linear = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        thickness = self.fc3(x)
        return thickness
    