import torch
from torch.nn.functional import softplus
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.means import ConstantMean
from gpytorch.utils.cholesky import psd_safe_cholesky

from voltron.models.BMGP import BMGP, MultitaskBMGP
from voltron.kernels import VolatilityKernel

# import sys
# sys.path.append("../means/")
from voltron.means import EWMAMean, DEWMAMean, TEWMAMean
from voltron.train_utils import LearnGPCV, TrainVolModel
from voltron.rollout_utils import Rollouts

class Volt(gpytorch.models.ExactGP):
    def __init__(self, train_x, log_data, mean='constant',
                 vol_path=None, k=25):
        
        # WE ASSUME IN THE BATCHED CASE THAT
        # TRAIN_X: N 
        # TRAIN_Y: T X N
        # VOL_PATH: T X N

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        super(Volt, self).__init__(train_x[1:], log_data[1:], likelihood)
        
        if log_data.ndim > 1:
            batch_shape = log_data.shape[:-1]
        else:
            batch_shape = torch.Size()
            
        if mean.lower() == 'constant':
            mean_module = gpytorch.means.ConstantMean().to(train_x.device)
        elif mean.lower() == 'ewma':
            mean_module = EWMAMean(train_x[1:], log_data[1:], k).to(train_x.device)
        elif mean.lower() == 'dewma':
            mean_module = DEWMAMean(train_x[1:], log_data[1:], k).to(train_x.device)
        elif mean.lower() == 'tewma':
            mean_module = TEWMAMean(train_x[1:], log_data[1:], k).to(train_x.device)
        else:
            print("ERROR: Mean not implemented")
            
        self.mean_module = mean_module.to(train_x.device)
        self.covar_module = VolatilityKernel().to(train_x.device)
        
        # but we store a T X N X 1 copy of train_x to maintain consistency w/ 
        # gpytorch
        if log_data.ndim > 1:
            self.train_x = train_x.unsqueeze(0).repeat(*batch_shape, 1)
        else:
            self.train_x = train_x
        self.train_y = log_data
        
        if vol_path is None:
            self.log_vol_path = -1 * torch.ones(train_x.shape[0]-1)
        else:
            self.log_vol_path = vol_path.log()
        
        self.train_cov = self.covar_module(self.train_x.unsqueeze(-1), self.log_vol_path.exp().unsqueeze(-1)).detach()
        
        if batch_shape == torch.Size():
            self.vol_lh = gpytorch.likelihoods.GaussianLikelihood()
            self.vol_model = BMGP(train_x, self.log_vol_path, self.vol_lh) 
        else:
            self.vol_lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=batch_shape[0])
            self.vol_lh.noise = 1e-3
            # we want the vol path GP to be N x T shaped and train_x to be N shaped
            self.vol_model = MultitaskBMGP(train_x, self.log_vol_path.t(), self.vol_lh)
        
    def UpdateVolPath(self, vol_path):
        self.log_vol_path = vol_path.log()
        self.train_cov = self.covar_module(self.train_inputs[0], self.log_vol_path.exp())
        return
        
    def VolMLL(self):
        vol_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.vol_lh, self.vol_model)
        outputs = self.vol_model(self.train_x)
        return vol_mll(outputs, self.log_vol_path)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        if torch.equal(x, self.train_inputs[0]):
            covar_x = self.train_cov
#             print("TRAIN COV")
        else:
            covar_x = self.covar_module(x, self.log_vol_path.exp())
#             print("NOT TRAIN COV")
            
#         print(covar_x.evaluate().shape)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def Train(self, gpcv_iters=400, vol_mod_iters=1000, data_mod_iters=400, display=False):
        x = self.train_x.squeeze()
        data = self.train_y.exp()

        ##############################
        ## Train GPCV and Vol Model ##
        ##############################
        vol = LearnGPCV(x[1:], data, gpcv_iters, printing=display)
        vmod, vlh = TrainVolModel(x[1:], vol, vol_mod_iters, printing=display)
        
        self.UpdateVolPath(vol)
        ######################
        ## Train Data Model ##
        ######################
        if isinstance(self.mean_module, (EWMAMean, DEWMAMean, TEWMAMean)):
            grad_flags = [True, False, False, False]
        else:
            grad_flags = [True, True, False, False, False]
            
        
        self.likelihood.raw_noise.data = torch.tensor([1e-5]).to(x.device)
        self.vol_lh = vlh.to(x.device)
        self.vol_model = vmod.to(x.device)
    
        for idx, p in enumerate(self.parameters()):
            p.requires_grad = grad_flags[idx]
            
        self.train();
        self.vol_lh.train();
        self.vol_model.train();
        
        
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        print_every = 50
        for i in range(data_mod_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(x[1:])
#             print(output)
#             print(data.log().shape)
            # Calc loss and backprop gradients
            loss = -mll(output, data.log()[1:])
            loss.backward()
            if display:
                if i % print_every == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, data_mod_iters, loss.item()))
            optimizer.step()
            
        
    def Forecast(self, test_x, nsample=50, return_vol=False, mean_revert=False, theta=0.05):
        self.vol_model.eval();
        self.eval();
        latent_mean = None
        if mean_revert:
            latent_mean = self.train_targets.squeeze().mean()
        samples = Rollouts(self.train_inputs[0].squeeze(),
                           self.train_targets.squeeze(), 
                           test_x, self, 
                            nsample=nsample,
                          return_vol=return_vol,
                          latent_mean=latent_mean, theta=theta)
        
        return samples