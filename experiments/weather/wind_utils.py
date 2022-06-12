import torch
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.cholesky import psd_safe_cholesky

def GenerateMeanRevertingPrediction(train_x, train_y, test_x, pred_vol, model, latent_mean=None, theta=0.5):
    vol = model.log_vol_path.exp()
    if model.train_x.ndim != test_x.ndim:
        test_x_for_stack = test_x.unsqueeze(0).repeat(model.train_x.shape[0], 1)
    else:
        test_x_for_stack = test_x
    if vol.ndim == 1:
        vol_for_stack = vol.unsqueeze(0).repeat(pred_vol.shape[0], 1)
    else:
        vol_for_stack = vol

    full_x = torch.cat((model.train_x, test_x_for_stack),dim=-1)
    # print("vol stack = ", vol_for_stack.shape)
    # print("pred_vol = ", pred_vol.shape)
    full_vol = torch.cat((vol_for_stack, pred_vol),dim=-1)
    
    test_x.repeat(2, test_x.numel())

    idx_cut = model.train_x.shape[-1]
    
    cov_mat = model.covar_module(full_x.unsqueeze(-1), full_vol.unsqueeze(-1)).evaluate()
    K_tr = cov_mat[..., :idx_cut, :idx_cut]
    K_tr_te = cov_mat[..., :idx_cut, idx_cut:]
    K_te = cov_mat[..., idx_cut:, idx_cut:]

    train_mean = model.mean_module(model.train_x)
    train_diffs = model.train_y.unsqueeze(-1) - train_mean.unsqueeze(-1)
    # use psd cholesky if you must evaluate
    K_tr_chol = psd_safe_cholesky(K_tr, jitter=1e-4)
    pred_mean = K_tr_te.transpose(-1, -2).matmul(torch.cholesky_solve(train_diffs, K_tr_chol))
    # print(voltron.mean_module(test_x).detach().T.shape)
    # print(pred_mean.shape)
    pred_mean += model.mean_module(test_x).detach().T.unsqueeze(-1)
    if latent_mean is not None:
        pred_mean -= theta * (pred_mean - latent_mean)

    pred_cov = K_te - K_tr_te.transpose(-1, -2).matmul(torch.cholesky_solve(K_tr_te, K_tr_chol))

    pred_cov_L = psd_safe_cholesky(pred_cov, jitter=1e-4)
    samples = torch.randn(*cov_mat.shape[:-2], test_x.shape[0], 1).to(test_x.device)
    samples = pred_cov_L @ samples

    if pred_mean.ndim == 1:
        return samples + pred_mean.unsqueeze(-1)
    else:
        return (samples + pred_mean).squeeze(-1)
    


def Rollouts(train_x, train_y, test_x, model, nsample=50, method = "volt", theta=0.5):
    if method != "volt":
        return nonvol_rollouts(train_x, train_y, test_x, model, nsample=nsample)
    latent_mean = train_y.log().mean()
    ntest = test_x.numel()
    samples = torch.zeros(nsample, ntest)
    pred_vol = model.vol_model(test_x).sample(torch.Size((nsample, ))).exp()
    samples[:, 0] = GeneratePrediction(train_x, train_y, 
                                       test_x[0].unsqueeze(0), 
                                       pred_vol[:, 0].unsqueeze(1),
                                      model, latent_mean, theta).squeeze()
    train_stack_y = train_y[1:].log().repeat(nsample, 1)
    train_stack_vol = model.log_vol_path.repeat(nsample, 1)
    
    for idx in range(1, ntest):
        stack_y = torch.cat((train_stack_y, 
                             samples[:, :idx].to(train_stack_y.device)), -1)
        stack_vol = torch.cat((train_stack_vol, 
                               pred_vol[:, :idx].to(train_stack_vol.device)), -1)
        rolling_x = torch.cat((train_x, test_x[:idx]))
        model.mean_module.train_y = stack_y
        model.mean_module.train_x = rolling_x
        
        model.train_x = rolling_x
        model.train_y = stack_y
        model.log_vol_path = stack_vol
        samples[:, idx] = GeneratePrediction(train_x, train_y, 
                                             test_x[idx].unsqueeze(0), 
                                           pred_vol[:, idx].unsqueeze(-1),
                                            model, latent_mean, theta).squeeze()
    return samples
