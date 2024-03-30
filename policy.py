import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParametricPolicy, self).__init__()
        # Define the architecture of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define the forward pass through the network
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#Uses tanh to limit the output between -1 and 1
class AffineThrottlePolicy(ParametricPolicy):

    def forward(self, x):
        x = super().forward(x)
        return torch.tanh(x)

# input_size = 8
# hidden_size = 128
# output_size = 2
#
# # Instantiate the policy
# policy = ParametricPolicy(input_size, hidden_size, output_size)
