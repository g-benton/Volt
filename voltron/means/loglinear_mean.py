import torch

from gpytorch.means import LinearMean

class LogLinearMean(LinearMean):
    def __init__(self, input_size, batch_shape=None, bias=True):
        if batch_shape is None:
            batch_shape = torch.Size()

        super().__init__(input_size=input_size, batch_shape=batch_shape, bias=bias)

    def initialize_from_data(self, x, y):
        with torch.no_grad():
            # assume y is on log scale
            self.bias.data = y.exp().mean(-1,keepdim=True)
            # is there anything we should do for the mean term?

    def forward(self, x):
        linear_term = super().forward(x)
        # to prevent linear stuff
        return linear_term.clamp(min=1e-6).log()