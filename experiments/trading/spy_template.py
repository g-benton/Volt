import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import os
import gpytorch
import argparse

from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import SpectralMixtureKernel, MaternKernel, RBFKernel, ScaleKernel

import sys
sys.path.append("../../magpie/means/")
from EWMA import EWMAMean, DEWMAMean, TEWMAMean

sys.path.append("../../magpie/")
from train_utils import LearnGPCV, TrainVolModel, TrainVoltMagpieModel, TrainBasicModel

sys.path.append("../../magpie/models/")
from VoltMagpie import VoltMagpie

from voltron.means import LogLinearMean
sys.path.append("../spdr-forecasting/")
from rollout_utils import GeneratePrediction, Rollouts

def main(args):
    savepath = "./saved-outputs/" + args.symbol + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    dpath = "../../magpie/data/"
    full_data = pd.read_csv(dpath + args.symbol + ".csv")
    
    full_data = pd.read_csv(dpath + args.symbol + ".csv")
    full_data["Date"] = pd.to_datetime(full_data['Date'])
    full_data = full_data.set_index(["Date"])
    
    ntrain = 450
    ntest = 20
    nsample = args.nsample
    train_iters = 500
    train_x = torch.arange(ntrain) * 1./252
    test_x = torch.arange(ntest) * 1./252 + train_x[-1] + train_x[1]
    dt = train_x[1] - train_x[0]
    
    start_date = pd.to_datetime("2011-11-21") - pd.Timedelta(ntrain, 'd')
    px = torch.FloatTensor(full_data.loc[start_date:].Close).squeeze()
    
    start_idxs = torch.arange(1, px.shape[0] - ntrain - ntest)
    
    ## save the typical stuff to use for plotting/validating ##
    torch.save({"ntrain":ntrain, "ntest":ntest, "start_idxs":start_idxs}, 
               "./saved-outputs/metadata.pt")
    
    if torch.cuda.is_available():
        use_cuda = True
        train_x = train_x.cuda()
        test_x = test_x.cuda()
    else:
        use_cuda = False
    

    fname = args.kernel + "_" + args.mean + str(args.k) + ".pt"

    save_samples = torch.zeros(start_idxs.numel(), nsample, ntest)
    for idx, start_idx in enumerate(start_idxs):
        train_y = px[start_idx-1:ntrain+start_idx].squeeze()
        test_y = px[start_idx + ntrain:start_idx + ntrain+ntest].squeeze()

        if use_cuda:
            train_y = train_y.cuda()
            test_y = test_y.cuda()
            test_x = test_x.cuda()

        if args.kernel.lower() == "volt":

            dt = train_x[1] - train_x[0]
            vol = LearnGPCV(train_x, train_y, train_iters=train_iters,
                            printing=False)
            vmod, vlh = TrainVolModel(train_x, vol, 
                                      train_iters=train_iters, printing=False)
            voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                                               vmod, vlh, vol,
                                               printing=False, 
                                               train_iters=train_iters,
                                               k=args.k, mean_func=args.mean)

            vmod.eval();

            if args.mean in ['ewma', 'dewma', 'tewma']:
                save_samples[idx, ::] = Rollouts(train_x, train_y, test_x, voltron, 
                                                 nsample=nsample)
            else: ## VOLT + STANDARD MEAN
                voltron.vol_model.eval()
                predvol = voltron.vol_model(test_x).sample(torch.Size((nsample, ))).exp()
                save_samples[idx, ::] = GeneratePrediction(train_x, train_y, test_x, 
                                             predvol, voltron).detach()
                del predvol
                
            del voltron, lh, vmod, vlh, vol
            
        else:
            kernel_possibilities = {"sm": SpectralMixtureKernel, "matern": MaternKernel, "rbf": RBFKernel}
            kernel = kernel_possibilities[args.kernel.lower()]
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
            mean_name = args.mean.lower()
            if mean_name == "loglinear":
                model.mean_module = LogLinearMean(1)
                model.mean_module.initialize_from_data(train_x, train_y.log())
            elif mean_name == 'linear':
                model.mean_module = LinearMean(1)
            elif mean_name == "constant":
                model.mean_module = ConstantMean()
            elif mean_name == "ewma":
                model.mean_module = EWMAMean(train_x, train_y.log(), k=args.k).to(train_x.device)
            elif mean_name == "dewma":
                model.mean_module = DEWMAMean(train_x, train_y.log(), k=args.k).to(train_x.device)
            elif mean_name == "tewma":
                model.mean_module = TEWMAMean(train_x, train_y.log(), k=args.k).to(train_x.device)

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
            
        print("Start Time = ", start_idx.item(), " out of ", len(start_idxs))
        torch.cuda.empty_cache()            
        torch.save(save_samples, savepath + fname)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
    )    
    parser.add_argument(
        "--kernel",
        type=str,
        default="volt",
    )    
    parser.add_argument(
        "--mean",
        type=str,
        default="ewma",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    main(args)
