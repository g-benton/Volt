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


def GenerateGPCVPredictions(ticker, dat, 
                        forecast_horizon=20, ntimes=25,
                        train_iters=400, nsample=1000,
                        ntrain=400):
    
    end_idxs = torch.arange(ntrain, dat.shape[0],
                           int((dat.shape[0]-ntrain)/ntimes))
    ntest = forecast_horizon
    dt = 1./252
    
    savepath = "./saved-outputs/" + ticker + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for last_day in end_idxs:
        date = str(dat.index[last_day.item()].date())
        print(date, ticker)
        train_y = torch.FloatTensor(dat.Close[last_day.item()-ntrain:last_day.item()].to_numpy())
        train_x = torch.arange(train_y.shape[0]-1) * dt
        test_x = torch.arange(ntest) * dt + train_x[-1] + train_x[1]
#         try:
        use_cuda = torch.cuda.is_available()
        if use_cuda: 
            train_x = train_x.cuda()
            test_x = test_x.cuda()
            train_y = train_y.cuda()
            
        model, likelihood = LearnGPCV(train_x, train_y,
                                train_iters=train_iters, printing=False, return_model=True)
        preds = likelihood(model(test_x),
                                return_gaussian=False).sample(torch.Size((nsample,)))
        preds = preds.cumsum(-1).squeeze()
        preds = preds.view(-1, preds.shape[-1])
        save_samples = preds.view(-1, preds.shape[-1]) * (dt ** 0.5) + train_y[-1].log()
        torch.save(save_samples, savepath + "gpcv_" + date + ".pt")

    return

def GenerateStockPredictions(ticker, dat, 
                        forecast_horizon=20,
                        train_iters=400, nsample=1000,
                        ntrain=400, mean='ewma', kernel='volt', 
                             save=False, k=300, ntimes=-1):
    
    if ntimes == -1:
        end_idxs = torch.arange(ntrain, dat.shape[0])
    else:
        end_idxs = torch.arange(ntrain, dat.shape[0],
                       int((dat.shape[0]-ntrain)/ntimes))
    ntest = forecast_horizon
    dt = 1./252
    
    model_name = kernel + "_" + mean + str(k) + "_"
    par_dir = "./saved-outputs/"
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)
    savepath = par_dir + ticker + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    for last_day in end_idxs:
        date = str(dat.index[last_day.item()].date())
#         try:
        train_y = torch.FloatTensor(dat.Close[last_day.item()-ntrain:last_day.item()].to_numpy())
        train_x = torch.arange(train_y.shape[0]-1) * dt
        test_x = torch.arange(ntest) * dt + train_x[-1] + train_x[1]
#         try:
        use_cuda = torch.cuda.is_available()
        if use_cuda: 
            train_x = train_x.cuda()
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
                save_samples = GeneratePrediction(train_x, train_y, test_x, 
                                             predvol, voltron).detach()
                del predvol

            del voltron, lh, vmod, vlh, vol
            torch.cuda.empty_cache()
        
        if save:
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            torch.save(save_samples, savepath + model_name + date + ".pt")
#         except:
#             nans = torch.ones(nsample, ntest) * torch.nan
#             if save:
#                 if not os.path.exists(savepath):
#                     os.mkdir(savepath)
#                 torch.save(nans, savepath + model_name + date + ".pt")
                

    return dat, save_samples



def GenerateOneDayPredictions(ticker, train_y, date,
                        forecast_horizon=20,
                        train_iters=400, nsample=1000,
                        ntrain=400, save=False, mean=None):
    
    ntest = forecast_horizon
    dt = 1./252
    par_dir = "./saved-outputs/"
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)
    savepath = par_dir + ticker + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    train_x = torch.arange(train_y.shape[0]-1) * dt
    test_x = torch.arange(ntest) * dt + train_x[-1] + train_x[1]
    use_cuda = torch.cuda.is_available()
    if use_cuda: 
        train_x = train_x.cuda()
        test_x = test_x.cuda()
        train_y = train_y.cuda()

    vol = LearnGPCV(train_x, train_y, train_iters=train_iters,
                        printing=False)
    vmod, vlh = TrainVolModel(train_x, vol,  
                              train_iters=train_iters, printing=False)
    
    if mean=='constant':
        voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                                                       vmod, vlh, vol,
                                                       printing=False, 
                                                       train_iters=200,
                                                       mean_func='constant')
        vmod.eval();
        voltron.eval();
        save_samples = Rollouts(train_x, train_y, test_x, voltron, 
                            nsample=nsample)
        
        if save:
            model_name = "volt_" + mean + "_"
            torch.save(save_samples, savepath + model_name + date + ".pt")
    else:
        for mean in ['ewma', 'dewma', 'tewma']:
            for k in [25, 50, 100, 200, 300, 400]:
                try:
                    voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                                                       vmod, vlh, vol,
                                                       printing=False, 
                                                       train_iters=0,
                                                       k=k, mean_func=mean)
                    vmod.eval();
                    voltron.eval();
                    save_samples = Rollouts(train_x, train_y, test_x, voltron, 
                                        nsample=nsample)
                except:
                    print("Failed: ", ticker, mean, k)
                    if save:
                        save_samples = torch.ones(nsample, ntest) * torch.nan

                if save:
                    model_name = "volt_" + mean + str(k) + "_"
                    torch.save(save_samples, savepath + model_name + date + ".pt")

        del voltron, lh, vmod, vlh, vol, save_samples
        torch.cuda.empty_cache()
    return



def GenerateBasicPredictions(ticker, dat, kernel_name, mean_name='ewma', k=400,
                        forecast_horizon=100,
                        train_iters=600, nsample=1000,
                        ntrain=400, save=False, ntimes=-1):
    
    if ntimes == -1:
        end_idxs = torch.arange(ntrain, dat.shape[0])
    else:
        end_idxs = torch.arange(ntrain, dat.shape[0],
                       int((dat.shape[0]-ntrain)/ntimes))
    ntest = forecast_horizon
    dt = 1./252
    par_dir = "./saved-outputs/"
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)
    savepath = par_dir + ticker + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for last_day in end_idxs:
        date = str(dat.index[last_day.item()].date())
        train_y = torch.FloatTensor(dat.Close[last_day.item()-ntrain:last_day.item()].to_numpy())
        train_x = torch.arange(train_y.shape[0]-1) * dt
        test_x = torch.arange(ntest) * dt + train_x[-1] + train_x[1]
#         try:
        use_cuda = torch.cuda.is_available()
        if use_cuda: 
            train_x = train_x.cuda()
            test_x = test_x.cuda()
            train_y = train_y.cuda()

#             print("Producing " + ticker + " Forecasts.....")
        kernel_possibilities = {"sm": SpectralMixtureKernel, 
                                "matern": MaternKernel, 
                                "rbf": RBFKernel}
        kernel = kernel_possibilities[kernel_name.lower()]
        if kernel_name.lower() != 'sm':
            kernel = ScaleKernel(kernel())
        else:
            kernel = kernel(num_mixtures=15)
            kernel.initialize_from_data_empspect(train_x, train_y.log())

        train_y = train_y[1:]

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

        if use_cuda:
            model = model.to(train_x.device)
        print("Fitting Model", ticker)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_torch(mll, options={'maxiter':train_iters, 'disp':False})
                
        if mean_name in ["loglinear", "constant", 'linear']:
            save_samples = model.posterior(test_x).sample(torch.Size((nsample,
                                                                    ))).squeeze(-1).cpu().detach()
        else:
            save_samples = Rollouts(
                train_x, train_y, test_x, model, nsample=nsample, method = "nonvol"
            ).cpu().detach()
            
        model_name = kernel_name + "_" + mean_name + str(k) + "_"
        torch.save(save_samples, savepath + model_name + date + ".pt")
            
        model.train()
        torch.cuda.empty_cache()
        del model
        

    return dat, save_samples