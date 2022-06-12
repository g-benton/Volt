import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import os
import gpytorch
import argparse
import datetime

from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import SpectralMixtureKernel, MaternKernel, RBFKernel, ScaleKernel
from voltron.means import EWMAMean, DEWMAMean, TEWMAMean
from voltron.train_utils import LearnGPCV, TrainVolModel, TrainVoltMagpieModel, TrainBasicModel
from voltron.models import VoltMagpie
from voltron.means import LogLinearMean

from voltron.rollout_utils import GeneratePrediction, Rollouts
from voltron.data import make_ticker_list, DataGetter, GetStockHistory


def BasicWindRollouts(train_x, train_y, test_x, kernel_name, mean_name='ewma', k=20,
                        train_iters=600, nsample=1000):
    

    kernel_possibilities = {"sm": SpectralMixtureKernel, 
                            "matern": MaternKernel, 
                            "rbf": RBFKernel}
    kernel = kernel_possibilities[kernel_name.lower()]
    if kernel_name.lower() != "sm":
        kernel = ScaleKernel(kernel())
    else:
        kernel = kernel(num_mixtures=20)
        kernel.initialize_from_data_empspect(train_x, train_y.log())

    model = SingleTaskGP(
        train_x.view(-1,1), 
        train_y.log().reshape(-1, 1), 
        covar_module=kernel, 
        likelihood=GaussianLikelihood()
    )

    mean_name = mean_name.lower()
    if mean_name == "loglinear":
        model.mean_module = LogLinearMean(1)
        model.mean_module.initialize_from_data(train_x, train_y.log())
    elif mean_name == 'linear':
        model.mean_module = LinearMean(1)
    elif mean_name == "constant":
        model.mean_module = ConstantMean()
    elif mean_name == "ewma":
        model.mean_module = EWMAMean(train_x, train_y.log(), k=k).to(train_x.device)
    elif mean_name == "dewma":
        model.mean_module = DEWMAMean(train_x, train_y.log(), k=k).to(train_x.device)
    elif mean_name == "tewma":
        model.mean_module = TEWMAMean(train_x, train_y.log(), k=k).to(train_x.device)

    
    model = model.to(train_x.device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_torch(mll, options={'maxiter':train_iters, 'disp':False})

    if mean_name in ["loglinear", "constant", 'linear']:
        save_samples = model.posterior(test_x).sample(torch.Size((nsample,
                                                                ))).squeeze(-1).cpu().detach()
    else:
        save_samples = Rollouts(
            train_x, train_y, test_x, model, nsample=nsample, method = "nonvol"
        ).cpu().detach()


    torch.cuda.empty_cache()
    del model


    return save_samples