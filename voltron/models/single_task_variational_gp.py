from typing import Union
from copy import deepcopy

import torch
import functools

from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior

from gpytorch import lazify
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import (
    CholLazyTensor,
    TriangularLazyTensor,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood as FNGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.utils.errors import NotPSDError
from gpytorch.utils.memoize import cached, add_to_cache, clear_cache_hook
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)

# from ..utils import pivoted_cholesky_init

# copied from wjmaddox/volatilitygp

# def _update_caches(m, *args, **kwargs):
#     if hasattr(m, "_memoize_cache"):
#         for key, item in m._memoize_cache.items():
#             if type(item) is not tuple and type(item) is not MultivariateNormal:
#                 if len(args) is 0:
#                     new_lc = item.to(torch.empty(0, **kwargs))
#                 else:
#                     new_lc = item.to(*args)
#                 m._memoize_cache[key] = new_lc
#                 if type(item) is TriangularLazyTensor:
#                     m._memoize_cache[key] = m._memoize_cache[key].double()
#             elif type(item) is MultivariateNormal:
#                 if len(args) is 0:
#                     new_lc = item.lazy_covariance_matrix.to(torch.empty(0, **kwargs))
#                 else:
#                     new_lc = item.lazy_covariance_matrix.to(*args)
#                 m._memoize_cache[key] = MultivariateNormal(
#                     item.mean.to(*args, **kwargs), new_lc
#                 )
#             else:
#                 m._memoize_cache[key] = (x.to(*args, **kwargs) for x in item)


# def _add_cache_hook(tsr, pred_strat):
#     if tsr.grad_fn is not None:
#         wrapper = functools.partial(clear_cache_hook, pred_strat)
#         functools.update_wrapper(wrapper, clear_cache_hook)
#         tsr.grad_fn.register_hook(wrapper)
#     return tsr


class _SingleTaskVariationalGP(ApproximateGP):
    def __init__(
        self,
        init_points: torch.Tensor = None,
        likelihood=None,
        learn_inducing_locations=True,
        covar_module=None,
        mean_module=None,
        use_piv_chol_init=True,
        num_inducing=None,
        use_whitened_var_strat=True,
        init_targets=None,
        train_inputs=None,
        train_targets=None,
    ):

        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel())

        inducing_points = init_points.detach().clone()

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.shape[-2]
        )
        if use_whitened_var_strat:
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        else:
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        super(_SingleTaskVariationalGP, self).__init__(variational_strategy)
        self.mean_module = ConstantMean() if mean_module is None else mean_module
        self.mean_module.to(init_points)
        self.covar_module = covar_module

        self.likelihood = GaussianLikelihood() if likelihood is None else likelihood
        self.likelihood.to(init_points)
        self.train_inputs = [train_inputs] if train_inputs is not None else [init_points]
        self.train_targets = train_targets if train_targets is not None else init_targets

        self.condition_into_exact = True

        self.to(init_points)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred

    # may actually want to keep this one in the future
    # def to(self, *args, **kwargs):
    #     _update_caches(self, *args, **kwargs)
    #     self.variational_strategy = self.variational_strategy.to(*args, **kwargs)
    #     _update_caches(self.variational_strategy, *args, **kwargs)
    #     return super().to(*args, **kwargs)


class SingleTaskVariationalGP(_SingleTaskVariationalGP, GPyTorchModel):
    def __init__(
        self,
        init_points=None,
        likelihood=None,
        learn_inducing_locations=True,
        covar_module=None,
        mean_module=None,
        use_piv_chol_init=True,
        num_inducing=None,
        use_whitened_var_strat=True,
        init_targets=None,
        train_inputs=None,
        train_targets=None,
        outcome_transform=None,
        input_transform=None,
    ):
        if outcome_transform is not None:
            is_gaussian_likelihood = (
                isinstance(likelihood, GaussianLikelihood) or likelihood is None
            )
            if train_targets is not None and is_gaussian_likelihood:
                if train_targets.ndim == 1:
                    train_targets = train_targets.unsqueeze(-1)
                train_targets, _ = outcome_transform(train_targets)

            if init_targets is not None and is_gaussian_likelihood:
                init_targets, _ = outcome_transform(init_targets)
                init_targets = init_targets.squeeze(-1)

        if train_targets is not None:
            train_targets = train_targets.squeeze(-1)

        # unlike in the exact gp case we need to use the input transform to pre-define the inducing pts
        if input_transform is not None:
            if init_points is not None:
                init_points = input_transform(init_points)

        _SingleTaskVariationalGP.__init__(
            self,
            init_points=init_points,
            likelihood=likelihood,
            learn_inducing_locations=learn_inducing_locations,
            covar_module=covar_module,
            mean_module=mean_module,
            use_piv_chol_init=use_piv_chol_init,
            num_inducing=num_inducing,
            use_whitened_var_strat=use_whitened_var_strat,
            init_targets=init_targets,
            train_inputs=train_inputs,
            train_targets=train_targets,
        )

        if input_transform is not None:
            self.input_transform = input_transform.to(
                self.variational_strategy.inducing_points
            )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform.to(
                self.variational_strategy.inducing_points
            )

    def forward(self, x):
        x = self.transform_inputs(x)
        return super().forward(x)

    @property
    def num_outputs(self) -> int:
        # we should only be able to have one output without a multitask variational strategy here
        return 1

    def initialize_variational_parameters(self, likelihood, x, f = None, y = None):
        #### also assumes inducing pts == train data, otherwise we do have to down project
        #### into the inducing space
        # TODO: f comes from newton iteration
        with torch.no_grad():
            # kuv = self.data_kernel(self.inducing_points, x)
            kuu = self.covar_module(self.variational_strategy.inducing_points)
            if f is None:
                assert y is not None
                # use unregularized version for now
                running_std = torch.stack([y[:i].std(0) for i in range(y.shape[0])])
                running_std[:10] = running_std[10]
                f = running_std.clamp(min=1e-4).log()
                # import matplotlib.pyplot as plt
                # plt.plot(f)
                # plt.show()
            
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
                kuu_chol.t().matmul(inverse_hessian).matmul(kuu_chol.evaluate())
            ).add_jitter(1.0)
            S = kuu_chol.matmul(inner_smat.inv_matmul(kuu_chol.t().evaluate()))

            self.variational_strategy._variational_distribution.variational_mean.data = f
            S_root = lazify(S).root_decomposition(method = "cholesky").root.evaluate().tril() * 10.
            self.variational_strategy._variational_distribution.chol_variational_covar.data = S_root.data
            self.variational_strategy.variational_params_initialized.fill_(1)

            self.mean_module.constant.data = running_std.mean(0).log()