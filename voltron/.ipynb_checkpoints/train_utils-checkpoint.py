import numpy as np
import torch
import gpytorch
import sys

sys.path.append("../")
from voltron.likelihoods import VolatilityGaussianLikelihood
from voltron.models import SingleTaskVariationalGP
from voltron.kernels import BMKernel, VolatilityKernel, FBMKernel
from voltron.models import BMGP, BasicGP, Volt
from voltron.means import LogLinearMean, EWMAMean, DEWMAMean, TEWMAMean, MeanRevertingEMAMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel


def LearnGPCV(train_x, train_y, train_iters=1000, printing=False, early_stopping=False, kernel = "bm"):
    dt = train_x[1]-train_x[0]
    scaled_returns = (train_y[1:] - train_y[:-1]) / (train_y[:-1]) / (dt**0.5)
    yy = scaled_returns
    
    likelihood = VolatilityGaussianLikelihood(param="exp")
    # likelihood.raw_a.data -= 4.
    if kernel == "bm":
        covar_module = BMKernel()
    elif kernel == "fbm":
        covar_module = FBMKernel()
    model = SingleTaskVariationalGP(
        init_points=train_x.view(-1,1), likelihood=likelihood, use_piv_chol_init=False,
        mean_module = gpytorch.means.ConstantMean(), covar_module=covar_module, 
        learn_inducing_locations=False, use_whitened_var_strat=False
    )
    model.initialize_variational_parameters(likelihood, train_x, y=yy)
    
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}, 
        # {"params": likelihood.parameters(), "lr": 0.1}
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, yy.numel(), combine_terms = True)
    
    print_every = 50
    for i in range(train_iters):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        with gpytorch.settings.num_gauss_hermite_locs(75):
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, yy)
            loss.backward()
            
            if printing:
                if i % print_every == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iters, loss.item()))
            optimizer.step()
    model.eval();
    likelihood.eval();
    predictive = model(train_x)
    pred_scale = likelihood(predictive, return_gaussian=False).scale.mean(0).detach()
    
    return pred_scale

def TrainVolModel(train_x, vol_path, train_iters=1000, printing=False, kernel = "bm"):
    vol_lh = gpytorch.likelihoods.GaussianLikelihood().to(train_x.device)
    vol_lh.noise.data = torch.tensor([1e-2])
    vol_model = BMGP(train_x, vol_path.log(), vol_lh, kernel=kernel).to(train_x.device)
#     vol_model.covar_module.raw_vol.data = torch.tensor([-3.])

    optimizer = torch.optim.Adam([
        {'params': vol_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(vol_lh, vol_model)
    
    print_every = 50
    for i in range(train_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = vol_model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, vol_path.log())
        loss.backward()
        if printing:
            if i % print_every == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iters, loss.item()))
        optimizer.step()
    return vol_model, vol_lh