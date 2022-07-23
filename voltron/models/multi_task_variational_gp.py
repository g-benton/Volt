import torch
import numpy as np

from gpytorch.models import GP
from gpytorch.kernels import IndexKernel
from gpytorch.lazy import KroneckerProductLazyTensor
from gpytorch.means import MultitaskMean, ConstantMean, ZeroMean
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch import lazify, settings

class MultitaskVariationalGP(GP):
    def __init__(self, inducing_points, num_tasks, covar_module = None, rank = 1, **kwargs):
        super().__init__()
        
        self.register_parameter(
            "variational_mean", torch.nn.Parameter(
                0.01 * torch.randn(inducing_points.shape[-1], num_tasks), requires_grad = True
            )
        )
        # TODO: parameterize the covariances as lower triangular only
        self.register_parameter(
            "variational_covar_root", torch.nn.Parameter(
                torch.eye(inducing_points.shape[-1]), requires_grad = True
            )
        )
        self.register_parameter(
            "variational_task_covar_root", torch.nn.Parameter(
                torch.eye(num_tasks), requires_grad = True
            )
        )
        
        self.index_kernel = IndexKernel(num_tasks=num_tasks, rank=rank, **kwargs)
        self.data_kernel = covar_module
        self.inducing_points = inducing_points
        self.num_tasks = num_tasks
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        
    def initialize_variational_parameters(self, likelihood, x, f = None, y = None):
        #### also assumes inducing pts == train data, otherwise we do have to down project
        #### into the inducing space
        # TODO: f comes from newton iteration
        with torch.no_grad():
            # kuv = self.data_kernel(self.inducing_points, x)
            kuu = self.data_kernel(self.inducing_points)
            if f is None:
                assert y is not None
                # use unregularized version for now
                running_std = torch.stack([y[:i].std(0) for i in range(y.shape[0])])
                running_std[:10] = running_std[10]
                f = running_std.clamp(min=1e-4).log()
            
            if likelihood.param == "exp":
                # this is the inverse hessian of the gp-exp parameterization
                inverse_hessian = torch.diag_embed(
                    (0.5 * y.pow(-2.0) * (f * 2.0).exp()).T
                ).clamp(min=1e-4, max=1000.)

            elif likelihood.param == "cv":
                y = f.t()
                f = ((y / likelihood.trans_a).exp() - 1 - likelihood.trans_c) / likelihood.trans_b
                
                sigma = likelihood(f.t()).scale.t()
                hessian_scaling = (2 + 3 * y.pow(2.0)) 
                second_deriv_scaling = likelihood.trans_a * likelihood.trans_b.pow(2.0) / 2
                scaling = (hessian_scaling * second_deriv_scaling).pow(-1.0)
                inverse_hessian = scaling * sigma.pow(2.0) * (1 + torch.cosh(likelihood.trans_b * y + likelihood.trans_c))
                inverse_hessian = torch.diag_embed(inverse_hessian)

                f = f.t()

            kuu_chol = kuu.cholesky()
            
            # this is the C parameterization of S
            # we not need to use the C, c parameterization for mu b/c the inducing pts are our training pts
            inner_smat = lazify(
                kuu_chol.t().matmul(inverse_hessian.mean(0)).matmul(kuu_chol.evaluate())
            ).add_jitter(1.0)
            inner_smat_inv_root = inner_smat.root_inv_decomposition().root.evaluate()
            S_root = kuu_chol.evaluate() @ inner_smat_inv_root

            self.variational_mean.data = f
            self.variational_covar_root.data = S_root * 10.
            log_means = running_std.clamp(min=1e-4).mean(0).log()
            [x.constant.data.add_(log_means[i]) for i, x in enumerate(self.mean_module.base_means)]
            
            if type(self.index_kernel) is IndexKernel:
                self.index_kernel.var.data /= 10.
                self.index_kernel.covar_factor.data /= 10.
        
    @property
    def variational_strategy(self):
        # hacky af for now
        return self
    
    def kl_divergence(self):
        ## this computes KL(q || p)
        
        prior_dist = MultitaskMultivariateNormal(
            self.mean_module(self.inducing_points).double(),
            KroneckerProductLazyTensor(
                self.data_kernel(self.inducing_points).double(), self.index_kernel.covar_matrix.double()
            )
        )

        Sxx = (self.variational_covar_root.tril() @ self.variational_covar_root.tril().transpose(-1, -2)).double()
        Stt = (self.variational_task_covar_root.tril() @ self.variational_task_covar_root.tril().t()).double()
        var_dist = MultitaskMultivariateNormal(
            self.variational_mean.double(),
            KroneckerProductLazyTensor(Sxx, Stt)
        )
        return torch.distributions.kl_divergence(var_dist, prior_dist).float()
    
    def forward(self, x, **kwargs):
        kuu = self.data_kernel(self.inducing_points)
        kux = self.data_kernel(self.inducing_points, x)
        kuu_inv_kux = kuu.inv_matmul(kux.evaluate())
        
        ### (Kxu Kuu^{-1} \otimes I)(m - \mu(z)) + \mu(t)
        inner_mean = self.variational_mean - self.mean_module(self.inducing_points)
        # the kronecker matmul is really a matrix matrix product
        mean_term = kuu_inv_kux.transpose(-1, -2).matmul(inner_mean)
        # add in mean term
        mean_term = mean_term + self.mean_module(x)
        
        ### covar is a sum of 3 terms
        ## T1 = Kxx \otimes I
        kxx = self.data_kernel(x)
        
        ## T2 = - (Kxu Kuu^{-1} Kux \otimes Ktt)
        data_onto_inducing = kux.transpose(-1, -2).matmul(kuu_inv_kux)

        first_covar_term = KroneckerProductLazyTensor(kxx - data_onto_inducing, self.index_kernel.covar_matrix)
        
        ## T3 = (Kxu Kuu^{-1} S Kuu^{-1} Kux \otimes I)
        variational_covar = self.variational_covar_root.tril().matmul(self.variational_covar_root.tril().T)
        variational_task_covar = self.variational_task_covar_root.tril().matmul(self.variational_task_covar_root.tril().T)
        data_onto_var_covar = kuu_inv_kux.transpose(-1, -2).matmul(
            variational_covar.matmul(kuu_inv_kux)
        )
        third_covar_term = KroneckerProductLazyTensor(data_onto_var_covar, variational_task_covar)
        
        ### Sigma = T1 + T2 + T3
        # returns a sumkroneckerlt so posterior sampling is effecient :)
        total_covar = first_covar_term + third_covar_term
        
        return MultitaskMultivariateNormal(mean_term, total_covar)