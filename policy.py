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

    def add_policy(self, other):
        """
        Adds the parameters of another ParametricPolicy instance to this one.
        Assumes both policies have the same architecture.
        """
        # Iterate through the parameters of both models
        for param_self, param_other in zip(self.parameters(), other.parameters()):
            # Add the parameters directly
            param_self.data += param_other.data

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
