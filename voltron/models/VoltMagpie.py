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

class VoltMagpie(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, vol_path=None,
                k=25):
        # WE ASSUME IN THE BATCHED CASE THAT
        # TRAIN_X: N 
        # TRAIN_Y: T X N
        # VOL_PATH: T X N

        super(VoltMagpie, self).__init__(train_x, train_y, likelihood)
        
        if train_y.ndim > 1:
            batch_shape = train_y.shape[:-1]
        else:
            batch_shape = torch.Size()
            
        self.mean_module = EWMAMean(train_x, train_y, k).to(train_x.device)
        self.covar_module = VolatilityKernel().to(train_x.device)
        
        # but we store a T X N X 1 copy of train_x to maintain consistency w/ 
        # gpytorch
        if train_y.ndim > 1:
            self.train_x = train_x.unsqueeze(0).repeat(*batch_shape, 1)
        else:
            self.train_x = train_x
        self.train_y = train_y
        
        if vol_path is None:
            self.log_vol_path = -1 * torch.ones(train_x.shape[0])
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
        self.train_cov = self.covar_module(self.train_x, self.log_vol_path.exp())
        return
        
    def VolMLL(self):
        vol_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.vol_lh, self.vol_model)
        outputs = self.vol_model(self.train_x)
        return vol_mll(outputs, self.log_vol_path)
    
    def GeneratePrediction(self, test_x, pred_vol, n_sample=1):
        if self.train_x.ndim != test_x.ndim:
            test_x_for_stack = test_x.unsqueeze(0).repeat(self.train_x.shape[0], 1)
        else:
            test_x_for_stack = test_x

        full_x = torch.cat((self.train_x, test_x_for_stack),dim=-1)
        full_vol = torch.cat((self.log_vol_path.exp(), pred_vol),dim=-1)

        idx_cut = self.train_x.shape[-1]
        cov_mat = self.covar_module(full_x.unsqueeze(-1), full_vol.unsqueeze(-1)).evaluate()
        
        K_tr = cov_mat[..., :idx_cut, :idx_cut]
        K_tr_te = cov_mat[..., :idx_cut, idx_cut:]
        K_te = cov_mat[..., idx_cut:, idx_cut:]

        train_mean = self.mean_module(*self.train_inputs).detach()
        train_diffs = self.train_y.unsqueeze(-1) - train_mean.unsqueeze(-1)
        
        # use psd cholesky if you must evaluate
        K_tr_chol = psd_safe_cholesky(K_tr)
        pred_mean = K_tr_te.transpose(-1, -2).matmul(torch.cholesky_solve(train_diffs, K_tr_chol))
        pred_mean += self.mean_module(test_x).detach().unsqueeze(-1)
                                                     
        pred_cov = K_te - K_tr_te.transpose(-1, -2).matmul(torch.cholesky_solve(K_tr_te, K_tr_chol))
        pred_cov_L = psd_safe_cholesky(pred_cov)
        samples = torch.randn(*cov_mat.shape[:-2], test_x.shape[0], n_sample)
        samples = pred_cov_L @ samples
        
        if pred_mean.ndim == 1:
            return samples + pred_mean.unsqueeze(-1)
        else:
            return (samples + pred_mean).squeeze(-1)
    
    def SamplePrediction(self, test_x, n_sample=1, return_vol=False):
        self.vol_model.eval()
        pred_vol = self.vol_model(test_x).sample().exp().transpose(-1, -2)
        
        prediction = self.GeneratePrediction(test_x, pred_vol, n_sample)
        if return_vol:
            return prediction, pred_vol
        else:
            return prediction
        
    def MeanPrediction(self, test_x, n_sample=1, return_vol=False):
        self.vol_model.eval();
        pred_vol = self.vol_model(test_x).mean.exp().transpose(-1, -2)
        prediction = self.GeneratePrediction(test_x, pred_vol, n_sample)
        if return_vol:
            return prediction, pred_vol
        else:
            return prediction
    
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        if torch.equal(x, self.train_inputs[0]):
            covar_x = self.train_cov
        else:
            covar_x = self.covar_module(x, self.log_vol_path.exp())
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)