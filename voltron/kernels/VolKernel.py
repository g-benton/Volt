import torch
from gpytorch.kernels import Kernel

def CumTrapz(y, x):
    dx = x[..., 1] - x[..., 0]
    dx = dx if x.ndim == 1 else dx.unsqueeze(-1)
    wghts = dx * torch.ones_like(x)
    wghts[..., 0] *= 0.5
    wghts[..., -1] *= 0.5
    return torch.cumsum(wghts * y, -1)

class VolatilityKernel(Kernel):
    has_lengthscale = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x, vol_path, diag=False, **params):    
        if x.shape[-1] == 1:
            x = x.squeeze()
        if vol_path.shape[-1] == 1:
            vol_path = vol_path.squeeze()

        last_dim_is_batch = params.get("last_dim_is_batch", False)
        if last_dim_is_batch:
            vol_path = vol_path.transpose(-1, -2)

        vol_int = CumTrapz(vol_path * vol_path, x)
        
        idx = torch.arange(x.shape[-1])
        idx1, idx2 = torch.meshgrid(idx, idx)
        idx = torch.minimum(idx1, idx2)
        res = vol_int[..., idx]

        # TODO: check this
        if last_dim_is_batch:
            res = res.permute(1, 2, 0)

        if diag:
            return torch.diagonal(res, dim1=-2, dim2=-1)
        else:
            return res