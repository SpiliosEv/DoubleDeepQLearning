import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 
import torch.nn.functional as F

class DQNNet_relu(nn.Module):
    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet_relu, self).__init__()
        self.dense1 = nn.Linear(input_size, 300)
        self.dense2 = nn.Linear(300, 300)
        self.dense3 = nn.Linear(300, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

