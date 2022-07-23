import torch
import gpytorch
from gpytorch.means import Mean
import numpy as np

def _EWMA(y, k):
    alpha = 2./(k + 1)
    conv = torch.nn.Conv1d(1, 1, kernel_size=k)
    wghts = alpha * (1-alpha)**(torch.arange(k-1, -1, -1))
    conv.weight.data = wghts.unsqueeze(0).unsqueeze(0)/wghts.sum()
    conv.bias.data = torch.zeros(1)

    padded_px = torch.cat((y.squeeze()[0] * torch.ones(k),
                           y.squeeze()))
    padded_px = padded_px.reshape(1, 1, -1)
    with torch.no_grad():
        ma = conv(padded_px).squeeze()
    return ma.type(torch.FloatTensor)

def EWMA(y, k):
    alpha = 2./(k + 1)
    conv = torch.nn.Conv1d(1, 1, kernel_size=k)
    wghts = alpha * (1-alpha)**(torch.arange(k-1, -1, -1))
    conv.weight.data = wghts.unsqueeze(0).unsqueeze(0)/wghts.sum()
    conv.bias.data = torch.zeros(1)    

    conv = conv.to(y.device)
    res = y[..., 0].unsqueeze(-1) * torch.ones(*y.shape[:-1], k).to(y.device)
    padded_px = torch.cat((res, y), dim=-1)
    batch_dim = y.shape[-2] if y.ndim > 1 else 1
    padded_px = padded_px.reshape(batch_dim, 1, -1)
    # print("padded_px shape = ", padded_px.shape)
    with torch.no_grad():
        ma = conv(padded_px).squeeze()
        
    # print("ma shape = ", ma.shape)
    return ma.type(torch.FloatTensor)

class EWMAMean(Mean):
    def __init__(self, train_x, train_y, k=20):
        super().__init__()
        self.k = k
        self.train_x = train_x
        self.train_y = train_y
        
    def forward(self, x):
        ewma = EWMA(self.train_y, self.k)
        if x.numel() == 1:
            res = ewma[..., -1].unsqueeze(0)
            return res.type(torch.FloatTensor).to(self.train_x.device)
        elif torch.equal(x.squeeze(), self.train_x.squeeze()):
            return ewma[..., :-1].type(torch.FloatTensor).to(self.train_x.device)
        else:
            return ewma.type(torch.FloatTensor).to(self.train_x.device)
        
        
class HEWMAMean(Mean):
    def __init__(self, train_x, train_y, k=20):
        super().__init__()
        self.k = k
        self.train_x = train_x
        self.train_y = train_y
        
    def forward(self, x):
        wma_k = EWMA(self.train_y, self.k)
        wma_k2 = EWMA(self.train_y, int(self.k/2))
        hma = EWMA(2*wma_k2[:-1] - wma_k[:-1], int(np.sqrt(self.k)))
        if torch.equal(x.squeeze(), self.train_x.squeeze()):
            return hma[:-1].type(torch.FloatTensor).to(self.train_x.device)
        else:
            return hma.type(torch.FloatTensor).to(self.train_x.device)
    
    
class DEWMAMean(Mean):
    def __init__(self, train_x, train_y, k=20):
        super().__init__()
        self.k = k
        self.train_x = train_x
        self.train_y = train_y
        
    def forward(self, x):
        ema = EWMA(self.train_y, self.k)#[..., :-1]
        ema_ema = EWMA(ema, self.k)[..., :-1]
        dema = 2*ema - ema_ema
        if x.numel() == 1:
            res = dema[..., -1].unsqueeze(0)
            return res.type(torch.FloatTensor).to(self.train_x.device)
        elif torch.equal(x.squeeze(), self.train_x.squeeze()):
            return dema[..., :-1].type(torch.FloatTensor).to(self.train_x.device)
        else:
            return dema.type(torch.FloatTensor).to(self.train_x.device)

        
class TEWMAMean(Mean):
    def __init__(self, train_x, train_y, k=20):
        super().__init__()
        self.k = k
        self.alpha = 2./(self.k + 1)
        self.train_x = train_x
        self.train_y = train_y
        
    def forward(self, x):
        ema = EWMA(self.train_y, self.k)
        ema_ema = EWMA(ema, self.k)[..., :-1]
        ema_ema_ema = EWMA(ema_ema, self.k)[..., :-1]
        tema = 3*ema - 3*ema_ema + ema_ema_ema
        if x.numel() == 1:
            res = tema[..., -1].unsqueeze(0)
            return res.type(torch.FloatTensor).to(self.train_x.device)
        elif torch.equal(x.squeeze(), self.train_x.squeeze()):
            return tema[..., :-1].type(torch.FloatTensor).to(self.train_x.device)
        else:
            return tema.type(torch.FloatTensor).to(self.train_x.device)