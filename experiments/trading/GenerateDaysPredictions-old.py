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
from voltron.data import make_ticker_list, DataGetter


def main(args):
    
    
    data_path = "../../voltron/data/"
    ticker_file = "test_tickers.txt"
    tckr_list = make_ticker_list(data_path + ticker_file)
    print("Downloading Data.....")
    
    if args.end_date.lower() == "none":
        end_date = str(datetime.date.today())
    else:
        end_date = args.end_date
    
    DataGetter(fpath = data_path, ticker_file=ticker_file, end_date=end_date)
    print("Data Downloaded.")
    use_cuda = torch.cuda.is_available()
    
    ntest = 20
    dt = 1./252
    
    print("Producing Forecasts.....")
    for tckr in tckr_list:
        dat = pd.read_csv(data_path + tckr + ".csv")
        train_x = torch.arange(dat.shape[0]-1) * dt
        test_x = torch.arange(ntest) * dt + train_x[-1] + train_x[1]
        train_y = torch.FloatTensor(dat.Close.to_numpy())

        if use_cuda: 
            train_x = train_x.cuda()
            test_x = test_x.cuda()
            train_y = train_y.cuda()

        train_iters=150
        mean = 'ewma'
        nsample = 1000
        if args.kernel == "volt":
            vol = LearnGPCV(train_x, train_y, train_iters=args.train_iters,
                                printing=False)
            vmod, vlh = TrainVolModel(train_x, vol, 
                                      train_iters=args.train_iters, printing=False)
            voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                                               vmod, vlh, vol,
                                               printing=False, 
                                               train_iters=args.train_iters,
                                               k=300, mean_func=args.mean)
            vmod.eval();
            if args.mean in ['ewma', 'dewma', 'tewma']:
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

        model_name = args.kernel + "_" + args.mean
        savepath = "./saved-outputs/" + tckr + "/"
        
        if not os.path.exists(savepath):
            os.mkdir(savepath)
            
        torch.save(save_samples, savepath + str(datetime.date.today()) + ".pt")
        if args.printing:
            print("\t" + tckr + " done.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--printing",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--end_date",
        default="none",
    )
    args = parser.parse_args()

    main(args)