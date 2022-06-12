import math
import torch
import gpytorch
import numpy as np
from voltron.means import EWMAMean, DEWMAMean, TEWMAMean
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from voltron.rollout_utils import nonvol_rollouts

class BasicGP():
    def __init__(self, train_x, train_y, kernel="matern", mean='constant', 
                 k=400, num_mixtures=10):
#         super(BasicGP, self).__init__(train_x, train_y, likelihood)
        if mean.lower() == 'constant':
            mean_module = gpytorch.means.ConstantMean().to(train_x.device)
        elif mean.lower() == 'ewma':
            mean_module = EWMAMean(train_x, train_y, k).to(train_x.device)
        elif mean.lower() == 'dewma':
            mean_module = DEWMAMean(train_x, train_y, k).to(train_x.device)
        elif mean.lower() == 'tewma':
            mean_module = TEWMAMean(train_x, train_y, k).to(train_x.device)
        else:
            print("ERROR: Mean not implemented")
        
        if kernel.lower() == 'matern':
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        elif kernel.lower() in ['sm', 'spectralmixture', 'spectral']:
            covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
            covar_module.initialize_from_data(train_x, train_y)
        elif kernel.lower() == 'rbf':
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            print("ERROR: Kernel not implemented")

        self.model = SingleTaskGP(train_x.view(-1, 1), train_y.reshape(-1, 1),
                                  covar_module=covar_module,
                                  likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.model.mean_module = mean_module
                
    def Train(self, train_iters=400, display=False):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_torch(mll, options={'maxiter':train_iters, 'disp':display})
        
    def Forecast(self, test_x, nsample=100):
        if not isinstance(self.model.mean_module, (EWMAMean, DEWMAMean, TEWMAMean)):
            
            samples = self.model.posterior(test_x).sample(torch.Size((nsample, )))
            
        else:
            samples = nonvol_rollouts(self.model.train_inputs[0].squeeze(),
                                      self.model.train_targets.squeeze(),
                                      test_x, self.model, nsample)
        return samples.squeeze()