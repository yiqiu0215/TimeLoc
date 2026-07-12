import torch.nn as nn


class TimeProj(nn.Module):
    """Linear -> ReLU -> Linear projection required by the design."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, inputs):
        return self.linear2(self.activation(self.linear1(inputs)))
