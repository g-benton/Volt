import torch
from gpytorch.kernels import Kernel

def CumTrapz(y, x):
    dx = x[1] - x[0]
    wghts = dx * torch.ones_like(x)
    wghts[0] *= 0.5
    wghts[-1] *= 0.5
    return torch.cumsum(wghts * y, 0)

class VolatilityKernel(Kernel):
    has_lengthscale = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x, vol_path, diag=False, **params):    
        
        last_dim_is_batch = params.get("last_dim_is_batch", False)
        if not last_dim_is_batch:
            vol_int = CumTrapz(vol_path.squeeze()**2, x.squeeze())
        else:
            x = x.unsqueeze(-1).repeat(x, vol_path.shape[-1])
            vol_int = CumTrapz(vol_path.pow(2.0), x)
        
        idx = torch.arange(x.shape[0])
        idx1, idx2 = torch.meshgrid(idx, idx)
        idx = torch.minimum(idx1, idx2)
        res = vol_int[idx]
            
        if vol_path.shape[-1] > 1:
            res = res.permute(2, 0, 1)
            
        if diag:
            return torch.diagonal(res, dim1=-2, dim2=-1)
        else:
            return res