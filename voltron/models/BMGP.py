from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import MultitaskKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from botorch.models import KroneckerMultiTaskGP

from voltron.kernels import BMKernel, FBMKernel

class BMGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel="bm"):
        super(BMGP, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = ConstantMean()
        if kernel == "bm":
            self.covar_module = BMKernel()
        elif kernel == "fbm":
            self.covar_module = FBMKernel()
            
        self.scaling = (train_x[1] - train_x[0])#.item()

    def mean_module(self, x):
        return -0.5 * self.covar_module.vol.pow(2.0) * x.squeeze()
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
#         if not self.training:
#         covar_x = covar_x * (self.scaling ** 0.5)
        return MultivariateNormal(mean_x, covar_x)
    
class MultitaskBMGP(KroneckerMultiTaskGP):
    def __init__(self, train_x, train_y, likelihood, base_mean=None, **kwargs):
        
        super(MultitaskBMGP, self).__init__(
            train_X=train_x.unsqueeze(-1), train_Y=train_y, likelihood=likelihood
        )
        self.covar_module = MultitaskKernel(BMKernel(), num_tasks=train_y.shape[-1], **kwargs)
        
        # init these smaller
        self.covar_module.task_covar_module.var.data /= 10.
        self.covar_module.task_covar_module.covar_factor.data /= 10.
        
        del self.mean_module
        
    def mean_module(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        scaled_mean = -0.5 * self.covar_module.data_covar_module.vol.pow(2.0) * \
            x.repeat(1, self.covar_module.num_tasks)
        return scaled_mean * self.covar_module.task_covar_module.covar_matrix.evaluate().diag()
        # need to take into acct intertask correlation here
        # return scaled_mean.matmul(self.covar_module.task_covar_module.covar_matrix.evaluate())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)    