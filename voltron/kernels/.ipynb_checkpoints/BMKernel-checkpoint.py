import torch
from torch.nn.functional import softplus
from gpytorch.kernels import Kernel

class BMKernel(Kernel):
    def __init__(self, vol=0., **kwargs):
        super(BMKernel, self).__init__(**kwargs)
        self.register_parameter(name='raw_vol', 
                                parameter=torch.nn.Parameter(vol*torch.ones(1)))
    
    def forward(self, x1s, x2s, **kwargs):        
        
        X1, X2 = torch.meshgrid(x1s[:, 0], x2s[:, 0])
#         return self.raw_vol.exp() * torch.minimum(X1,X2)
        cov = self.raw_vol.exp() * torch.minimum(X1,X2)
        return cov