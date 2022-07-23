import torch

from gpytorch.means import Mean
from gpytorch.utils.broadcasting import _mul_broadcast_shape

class MulIdentityMean(Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        super(MulIdentityMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        # if input.shape[:-2] == self.batch_shape:
        #     exp_const = self.constant.expand(input.shape[:-1])
        # else:
        #     exp_const = self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))
        return (self.constant * input).squeeze(-1)