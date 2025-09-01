import torch 
import torch.nn as nn

"""model置き場"""

class ThreeFullyConnectedLayers(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden1_dim = 64,
                 hidden2_dim = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    