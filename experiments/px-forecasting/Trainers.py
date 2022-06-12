import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gpytorch
# from voltron.robinhood_utils import GetStockData
import os
# import robin_stocks.robinhood as r
import pickle5 as pickle

sns.set_style("whitegrid")
sns.set_palette("bright")

sns.set(font_scale=2.0)
sns.set_style('whitegrid')

import sys
sys.path.append("../")
from voltron.likelihoods import VolatilityGaussianLikelihood
from voltron.models import SingleTaskVariationalGP as SingleTaskCopulaProcessModel
from voltron.kernels import BMKernel, VolatilityKernel
from voltron.models import BMGP, VoltronGP, MaternGP, SMGP
from voltron.means import LogLinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel

def get_and_fit_gpcv(x, log_returns, printing=False):
    train_x = x[:-1]
    dt = train_x[1]-train_x[0]
    # prepare model
    likelihood = VolatilityGaussianLikelihood(param="exp")
    # likelihood.raw_a.data -= 6.
    covar_module = BMKernel()
    model = SingleTaskCopulaProcessModel(
        init_points=train_x.view(-1,1), 
        likelihood=likelihood, 
        use_piv_chol_init=False,
        mean_module = gpytorch.means.ConstantMean(), 
        covar_module=covar_module, 
        learn_inducing_locations=False
    )
    model.mean_module.constant.data -= 4.
    # model.initialize_variational_parameters(likelihood, train_x, y=log_returns)
    
    import os
    smoke_test = ('CI' in os.environ)
    training_iterations = 2 if smoke_test else 500


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # likelihood parameters should be taken acct of in the model
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}, 
        # {"params": likelihood.parameters(), "lr": 0.1}
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, log_returns.numel())
    
    old_loss = 10000.
    print_every = 50
    for i in range(training_iterations):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        with gpytorch.settings.num_gauss_hermite_locs(75):
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, log_returns)
            loss.backward()
            if printing:
                if i % print_every == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, 
                                                       training_iterations, 
                                                       loss.item()))
            optimizer.step()
        if old_loss <= loss and i > 100:
            if printing:
                print(old_loss, loss)
            break
        else:
            old_loss = loss.item()
            
    model.eval();
    likelihood.eval();
    predictive = model(x)
    pred_scale = likelihood(predictive).scale.mean(0).detach()
    samples = likelihood(predictive).scale.detach()
    
#     plt.plot(x, pred_scale, linewidth = 4)
#     plt.plot(x, samples.t(), color = "gray", alpha = 0.3)
#     # plt.ylim((0, 0.25))
#     plt.show()
    
    # return scaled volatility prediction
    return pred_scale / dt**0.5

def get_and_fit_vol_model(train_x, est_vol):
    vol_lh = gpytorch.likelihoods.GaussianLikelihood()
    vol_lh.noise.data = torch.tensor([1e-6])
    vol_model = BMGP(train_x, est_vol.log(), vol_lh)

    optimizer = torch.optim.Adam([
        {'params': vol_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(vol_lh, vol_model)
    old_loss = 10000
    for i in range(500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = vol_model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, est_vol.log())
        loss.backward()
        if i % 50 == 0:
            print(loss.item())
        optimizer.step()
#         if old_loss <= loss:
#             break
#         else:
#             old_loss = loss.item()
        
    return vol_model

def get_and_fit_data_model(train_x, train_y, pred_vol, vol_model):
    voltron_lh = gpytorch.likelihoods.GaussianLikelihood()
    voltron = VoltronGP(train_x, train_y.log(), voltron_lh, pred_vol)
    # voltron.mean_module = gpytorch.means.LinearMean(1)
    voltron.mean_module = LogLinearMean(1)
    voltron.mean_module.initialize_from_data(train_x, train_y.log())
    voltron.likelihood.raw_noise.data = torch.tensor([1e-6])
    voltron.vol_lh = vol_model.likelihood
    voltron.vol_model = vol_model

    grad_flags = [False, True, True, True, False, False, False]

    for idx, p in enumerate(voltron.parameters()):
        p.requires_grad = grad_flags[idx]

    voltron.train();
    voltron_lh.train();
    voltron.vol_lh.train();
    voltron.vol_model.train();

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': voltron.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(voltron_lh, voltron)

    for i in range(500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = voltron(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.log())
        loss.backward()
        # print(loss.item())
        optimizer.step()
    return voltron

def predict_prices(test_x, voltron, nvol=10, npx=10):
    ntest = test_x.shape[0]
    vol_paths = torch.zeros(nvol, ntest)
    px_paths = torch.zeros(npx*nvol, ntest)

    voltron.vol_model.eval();
    voltron.eval();

    for vidx in range(nvol):
        vol_pred = voltron.vol_model(test_x).sample().exp()
        vol_paths[vidx, :] = vol_pred.detach()

        px_pred = voltron.GeneratePrediction(test_x, vol_pred, npx).exp()
        px_paths[vidx*npx:(vidx*npx+npx), :] = px_pred.detach().T
    return px_paths

def get_and_fit_basic_model(train_x, train_y, cov="matern", mean="loglinear"):
    voltron_lh = gpytorch.likelihoods.GaussianLikelihood()
#     voltron = VoltronGP(train_x, train_y.log(), voltron_lh, pred_vol)
    if cov == "matern":
        model = MaternGP(train_x, 
                             train_y.log(), likelihood=voltron_lh)
    else:
        model = SMGP(train_x, 
                             train_y.log(), likelihood=voltron_lh)
    if mean == "loglinear":
        model.mean_module = LogLinearMean(1)
        model.mean_module.initialize_from_data(train_x, train_y.log())
    else:
        model.mean_module = gpytorch.means.ConstantMean()
        
    
    model.likelihood.raw_noise.data = torch.tensor([1e-6])

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(voltron_lh, model)

    for i in range(500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.log())
        loss.backward()
        # print(loss.item())
        optimizer.step()
    return model, voltron_lh

def predict_basic_prices(test_x, voltron, voltron_lh, npath=1000):
    ntest = test_x.shape[0]
    voltron.eval();
    mod = voltron_lh(voltron(test_x))
    px_paths = mod.sample(torch.Size(((npath),))).exp().squeeze(-1)
    
    return px_paths
