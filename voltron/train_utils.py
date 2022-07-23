import numpy as np
import torch
import gpytorch
import sys

sys.path.append("../")
from voltron.likelihoods import VolatilityGaussianLikelihood
from voltron.models import SingleTaskVariationalGP
from voltron.kernels import BMKernel, VolatilityKernel, FBMKernel
from voltron.models import BMGP, VoltronGP, MaternGP, SMGP, VoltMagpie
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
    
    old_loss = 10000.
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


def TrainDataModel(train_x, train_y, vol_model, vol_lh, vol_path,
                   train_iters=1000, printing=False):
    voltron_lh = gpytorch.likelihoods.GaussianLikelihood()
    voltron = VoltronGP(train_x, train_y.log(), voltron_lh, vol_path)
    voltron.mean_module = LogLinearMean(1)
    # voltron.mean_module.register_prior("slope_prior", 
    #         gpytorch.priors.NormalPrior(0,0.1), 'weights')
    voltron.mean_module.initialize_from_data(train_x, train_y.log())
    # voltron.mean_module.weights.data = torch.tensor([[0.]])
    voltron.likelihood.raw_noise.data = torch.tensor([1e-5])
    voltron.vol_lh = vol_lh
    voltron.vol_model = vol_model

    grad_flags = [True, True, True, False, False, False]

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

    print_every = 50
    for i in range(train_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = voltron(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.log())
        loss.backward()
        if printing:
            if i % print_every == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iters, loss.item()))
        optimizer.step()
        
        
    return voltron, voltron_lh

def TrainBasicModel(train_x, train_y, train_iters=1000, printing=False, 
                    model_type="matern",
                   num_mixtures=10, mean_func="loglinear"):
    lh = gpytorch.likelihoods.GaussianLikelihood()
    
    if model_type == "matern":
        model = MaternGP(train_x, train_y.log(), lh)
    else:
        model = SMGP(train_x, train_y.log(), lh, num_mixtures)
    
    if mean_func == "loglinear":
        model.mean_module = LogLinearMean(1)
        model.mean_module.register_prior("slope_prior", 
                gpytorch.priors.NormalPrior(0,0.1), 'weights')
        model.mean_module.initialize_from_data(train_x, train_y.log())
    
    model.likelihood.raw_noise.data = torch.tensor([1e-5])
    model = model.to(train_x.device)
    model.train();
    lh.train();

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)
    print_every = 50
    for i in range(train_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.log())
        loss.backward()
        if printing:
            if i % print_every == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iters, loss.item()))
        optimizer.step()
        
        
    return model, lh


def TrainVoltMagpieModel(train_x, train_y, vol_model, vol_lh, vol_path,
                   train_iters=1000, printing=False, k=25, theta=0.5,
                        mean_func="ewma"):
    voltron_lh = gpytorch.likelihoods.GaussianLikelihood().to(train_x.device)
    voltron = VoltMagpie(train_x, train_y.log(),
                             voltron_lh, vol_path, k=k).to(train_x.device)
    
    if mean_func.lower() in ["ewma", "dewma", "tewma", "meanrevert"]:
        # default voltmagpie is an ewma mean so we don't need to redefine anything
        grad_flags = [True, False, False, False]
        
        if mean_func.lower() == "dewma":
            voltron.mean_module = DEWMAMean(train_x, train_y.log(), k).to(train_x.device)
        elif mean_func.lower() == 'tewma':
            voltron.mean_module = TEWMAMean(train_x, train_y.log(), k).to(train_x.device)
        elif mean_func.lower() == 'meanrevert':
            voltron.mean_module = MeanRevertingEMAMean(train_x, train_y.log(),
                                                      k, theta).to(train_x.device)
            
    elif mean_func.lower()=='constant':
        voltron.mean_module = gpytorch.means.ConstantMean().to(train_x.device)
        grad_flags = [True, True, False, False, False]
    elif mean_func.lower()=='loglinear':
        voltron.mean_module = LogLinearMean(1).to(train_x.device)
        voltron.mean_module.initialize_from_data(train_x, train_y.log())
        grad_flags = [True, True, True, False, False, False]
    elif mean_func.lower()=='linear':
        voltron.mean_module = gpytorch.means.LinearMean(1).to(train_x.device)
        grad_flags = [True, True, True, False, False, False]

    voltron.likelihood.raw_noise.data = torch.tensor([1e-5]).to(train_x.device)
    voltron.vol_lh = vol_lh.to(train_x.device)
    voltron.vol_model = vol_model.to(train_x.device)

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

    print_every = 50
    for i in range(train_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = voltron(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.log())
        loss.backward()
        if printing:
            if i % print_every == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iters, loss.item()))
        optimizer.step()
        
        
    return voltron, voltron_lh