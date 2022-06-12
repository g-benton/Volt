import numpy as np
import torch
import pandas as pd
import copy
from voltron.option_utils import GetTradingDays, GetTrainingData, Pricer, FindLastTradingDays
from scipy.optimize import minimize

def BlackVol(pars, K, f, T):
    alpha = torch.exp(pars[0][0]) ## v0
    rho = 2 * torch.sigmoid(pars[0][1]) - 1. ##rho
    v = torch.exp(pars[0][2]) ## "sigma" 
    beta = 1.
    num = 1 + (alpha**2 * (1-beta)**2/(24 * (f*K)**(1-beta)) +\
               0.25 * rho*beta*v*alpha/((f*K)**(0.5*(1-beta))) +\
               v**2*(2-3*rho**2)/24)*T
    num*= alpha
    
    denom = (f*K)**(0.5*(1-beta)) * (1 + (1-beta)**2/24 * torch.log(f/K)**2 +\
                                     (1-beta)**4/1920 * torch.log(f/K)**4)
    
    z = v/alpha * (f*K)**(0.5*(1-beta)) * np.log(f/K)
    xi_z = torch.log((torch.sqrt(1 - 2 * rho * z + z**2) + z - rho)/(1-rho))
    
    return num/denom * z/xi_z

def MinVol(pars, Ks, Fs, Ts, ivol):
    return torch.mean((ivol - BlackVol(pars, Ks, Fs, Ts)).pow(2))

def Calibrate(Fs, Ks, Ts, ivol, iters=1000):
    pars = [torch.tensor([-1., -5., -3.], requires_grad=True)]
    opt = torch.optim.SGD(pars, lr=0.1)
    stored_pars = torch.zeros(iters, 3)
    losses = []
    for e in range(iters):
        stored_pars[e, :] = pars[0]
        loss = MinVol(pars, Ks, Fs, Ts, ivol)
        opt.zero_grad()
        loss.backward()
        losses.append(loss.item())
        opt.step() 
    
    return pars[0].detach().numpy()

def SABRSim(Np, Nt, S0, V0, sigma, rho, dt=1./252.):
    dW = np.random.randn(Nt+1, Np) * np.sqrt(dt)
    dZ = rho * dW + np.sqrt(1-rho**2) * np.random.randn(Nt+1, Np) * np.sqrt(dt)
    
    S = np.zeros((Nt+1, Np))
    S[0] = S0
    V = np.zeros((Nt+1, Np))
    V[0] = V0
    
    for t in range(Nt):
        S[t+1] = S[t] + V[t]*S[t]*dW[t]
        V[t+1] = V[t] + sigma*V[t]*dZ[t]
        
    return S[1:]

def main():
    years = [yr for yr in range(2006, 2018)]
    logger = []
    full_logger = []
    SPY = pd.read_csv("./data/SPY_prices.csv")
    SPY['Date'] = pd.to_datetime(SPY['Date'])
    Np = 10000
    for year in years:
        options = pd.read_csv("./data/SPY_" + str(year) + ".csv")
        options.expiration = pd.to_datetime(options.expiration)
        options.quotedate = pd.to_datetime(options.quotedate)
        qday = options.quotedate.unique()[0]
        quote_price = SPY[SPY['Date']==qday].Close.item()
        options = options[(options.quotedate == qday) & (options.type=='call')]
        edays = options.expiration.sort_values().unique()
        testdays = (edays - qday)/np.timedelta64(1, "D")
        edays = edays[(testdays > 100) & (testdays < 365)]
        lastdays = FindLastTradingDays(SPY, edays)
        ntests = np.array([GetTradingDays(SPY, qday, 
                                          pd.Timestamp(ld)) for ld in lastdays])
        fulltest = ntests[-1]
        
        test_y = torch.FloatTensor(GetTrainingData(SPY, 
                                           pd.Timestamp(lastdays[-1]),
                                           fulltest).to_numpy())
    
        ## extract data for calibration ##
        ivol = torch.tensor(options.impliedvol.to_numpy())
        Fs = torch.tensor(options.underlying_last.to_numpy())
        Ks = torch.tensor(options.strike.to_numpy())
        starts = options.quotedate.dt.date.to_numpy()
        ends = options.expiration.dt.date.to_numpy()
        Ts = torch.tensor(([np.busday_count(qd, ed)/252. for qd, ed in zip(starts, ends)]))
        
        pars = Calibrate(Fs, Ks, Ts, ivol)
        v0 = np.exp(pars[0])
        1/(1 + np.exp(-pars[1]))
        rho = (2/(1 + np.exp(-pars[1])) - 1.)
        sigma = np.exp(pars[2])
        px_paths = SABRSim(Np, fulltest, quote_price, v0, sigma, rho)
        # px_samples = torch.tensor(px_paths[ntests-1])
        
        option_output = Pricer(torch.tensor(px_paths), options, edays, test_y[ntests-1],
                               quote_price)
        
        option_output.to_pickle("./output/sabr" + str(year) + ".pkl")
        print(str(year), "Done")
        
    
if __name__ == "__main__":
    main()
    