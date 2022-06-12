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


def GenerateStockPredictions(ticker, dat, 
                        forecast_horizon=20,
                        train_iters=400, nsample=1000,
                        ntrain=400, mean='ewma', kernel='volt', 
                             save=False, k=300):
    
    end_idxs = torch.arange(ntrain, dat.shape[0])
    ntest = forecast_horizon
    dt = 1./252
    
    model_name = kernel + "_" + mean + str(k) + "_"
    savepath = "./saved-outputs/" + ticker + "/"
    
    for last_day in end_idxs:
        date = str(dat.index[last_day.item()].date())
        try:
            train_y = torch.FloatTensor(dat.Close[last_day.item()-ntrain:last_day.item()].to_numpy())
            train_x = torch.arange(train_y.shape[0]-1) * dt
            test_x = torch.arange(ntest) * dt + train_x[-1] + train_x[1]
    #         try:
            use_cuda = torch.cuda.is_available()
            if use_cuda: 
                train_x = trin_x.cuda()
                test_x = test_x.cuda()
                train_y = train_y.cuda()

    #             print("Producing " + ticker + " Forecasts.....")
            if kernel == "volt":
                vol = LearnGPCV(train_x, train_y, train_iters=train_iters,
                                    printing=False)
                vmod, vlh = TrainVolModel(train_x, vol, 
                                          train_iters=train_iters, printing=False)
                voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                                                   vmod, vlh, vol,
                                                   printing=False, 
                                                   train_iters=train_iters,
                                                   k=k, mean_func=mean)
                vmod.eval();
                if mean in ['ewma', 'dewma', 'tewma']:
                    save_samples = Rollouts(train_x, train_y, test_x, voltron, 
                                            nsample=nsample)

                else: ## VOLT + STANDARD MEAN
                    voltron.vol_model.eval()
                    predvol = voltron.vol_model(test_x).sample(torch.Size((nsample, ))).exp()
                    save_samples[idx, ::] = GeneratePrediction(train_x, train_y, test_x, 
                                                 predvol, voltron).detach()
                    del predvol

                del voltron, lh, vmod, vlh, vol
                torch.cuda.empty_cache()
            else:
                kernel_possibilities = {"sm": SpectralMixtureKernel, "matern": MaternKernel, "rbf": RBFKernel}
                kernel = kernel_possibilities[kernel.lower()]
                if type(kernel) is not SpectralMixtureKernel:
                    kernel = ScaleKernel(kernel())
                else:
                    kernel = kernel()
                    kernel.initialize_from_data_empspect(train_x, train_y.log())

                train_y = train_y[1:]

                model = SingleTaskGP(
                    train_x.view(-1,1), 
                    train_y.log().reshape(-1, 1), 
                    covar_module=kernel, 
                    likelihood=GaussianLikelihood()
                )
                mean_name = mean.lower()
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

                if use_cuda:
                    model = model.to(train_x.device)

                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_torch(mll, options={'maxiter':train_iters, 'disp':False})

                if mean_name in ["loglinear", "constant", 'linear']:
                    save_samples[idx] = model.posterior(test_x).sample(torch.Size((nsample, ))).squeeze(-1).cpu().detach()
                else:
                    save_samples[idx] = Rollouts(
                        train_x, train_y, test_x, model, nsample=nsample, method = "nonvol"
                    ).cpu().detach()

                torch.cuda.empty_cache()
                del model, kernel

            if save:
                if not os.path.exists(savepath):
                    os.mkdir(savepath)

                torch.save(save_samples, savepath + model_name + date + ".pt")
        except:
            nans = torch.ones(nsample, ntest) * torch.nan
            if save:
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                torch.save(nans, savepath + model_name + date + ".pt")
                

    return dat, save_samples