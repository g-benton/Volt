import torch
import gpytorch
from voltron.likelihoods import VolatilityGaussianLikelihood
from voltron.kernels import BMKernel, VolatilityKernel
from voltron.models import BMGP, VoltronGP
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from voltron.models import MultitaskVariationalGP
from gpytorch.priors import LKJCovariancePrior, SmoothedBoxPrior
from voltron.models import MultitaskBMGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood

def get_and_fit_volmodel(train_x, pred_scale, train_iter=200,
                        printing=False):
    prior = LKJCovariancePrior(eta=5.0, n=pred_scale.shape[-1], sd_prior=SmoothedBoxPrior(0.05, 1.0))

    vol_lh = MultitaskGaussianLikelihood(num_tasks=pred_scale.shape[-1])
    vol_lh.noise.data = torch.tensor([1e-6])
    vol_model = MultitaskBMGP(train_x, pred_scale.log(), vol_lh, prior=prior).to(train_x.device)

    optimizer = torch.optim.Adam([
        {'params': vol_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(vol_lh, vol_model)

    for i in range(train_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = vol_model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, pred_scale.log())
        loss.backward()
        if printing:
            if i % 50 == 0:
                print(loss.item(), vol_model.covar_module.data_covar_module.raw_vol.item())
        optimizer.step()
        
    
    return vol_model, vol_lh

def get_and_fit_mtgpcv(train_x, log_returns, train_iter=200, printing=False):
    likelihood = VolatilityGaussianLikelihood(batch_shape=[log_returns.shape[0]], param="exp")
    dt = train_x[1] - train_x[0]
    # corresponds to ICM
    model = MultitaskVariationalGP(
        inducing_points=train_x, 
        covar_module=BMKernel().to(train_x.device), learn_inducing_locations=False,
        num_tasks = log_returns.shape[0], 
        prior=LKJCovariancePrior(eta=5.0, n=log_returns.shape[0], sd_prior=SmoothedBoxPrior(0.05, 1.0))
    )
    model = model.to(train_x.device)
    model.initialize_variational_parameters(likelihood=likelihood, x=train_x, y=log_returns.t())
    
    model = model.to(train_x.device)
    likelihood = likelihood.to(train_x.device)


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}, 
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_x.shape[0])
    
    batched_train_x = train_x#[:-1]
    
    print_every = 50
    for i in range(train_iter):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(batched_train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, log_returns.t())
        loss.backward()
        if printing:
            if i % print_every == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iter, loss.item()))
        optimizer.step()
        
    model.eval();
    likelihood.eval();
    predictive = model(train_x)
    # pred_scale = likelihood(predictive).scale.mean(0).detach()
    samples = likelihood(predictive).scale.detach()
    return samples.mean(0) / dt**0.5



def get_and_fit_datamods(train_x, train_y, vols, vmod, vlh,
                         train_iter=200, printing=False, k=200, mean_func='ewma'):
    model_list = []
    nmodel = vols.shape[-1]
    for mdl_idx in range(nmodel):
        mod = TrainVoltMagpieModel2(train_x, train_y[mdl_idx, :], vmod, vlh, vols[:, mdl_idx],
                                  train_iters=train_iter, k=k, mean_func=mean_func)
        model_list.append(mod)
        
    return model_list
    

def TrainVoltMagpieModel2(train_x, train_y, vol_model, vol_lh, vol_path,
                   train_iters=1000, printing=False, k=25,
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
#     voltron.vol_lh = vol_lh.to(train_x.device)
#     voltron.vol_model = vol_model.to(train_x.device)
    
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