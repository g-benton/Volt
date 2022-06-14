import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import seaborn as sns
import time
import copy
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable  

def NLL(targets, outputs):
    dist = torch.distributions.Normal(outputs[:, 0], outputs[:, 1])
    return -dist.log_prob(targets).sum()

class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length=5):
        self.sequence_length = sequence_length
        self.X = data.float()

    def __len__(self):
        return self.X.shape[0]-1

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1)]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1).squeeze(-1)
            x = self.X[0:(i + 1)]
            x = torch.cat((padding, x), 0)
            
        return x.unsqueeze(0), self.X[i+1]

class LSTM(nn.Module):
    def __init__(self, num_classes, seq_len, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = seq_len #input size
        self.hidden_size = hidden_size #hidden state

        self.lstm = nn.LSTM(input_size=seq_len, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state

        hn = hn[self.num_layers-1]
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        
        output = torch.zeros_like(out)
        output[:, 0] = out[:, 0]
        output[:, 1] = self.softplus(out[:, 1])
        return output
    
def TrainLSTM(data_loader, model, loss_function, optimizer, epochs=200,
             printing=False, use_cuda=False):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    for epoch in range(epochs):
        for X, y in data_loader:
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            output = model(X)
            loss = loss_function(y, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if printing:
            if epoch%10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Train loss: {avg_loss}, Epoch: {epoch}")
    
def LSTMRollouts(model, nrollout, rollout_len, dset, use_cuda=False):
    xin, xout = dset[len(dset)-1]
    xx = torch.cat((xin[0, 1:], xout.unsqueeze(0)))
    xx = xx.repeat(nrollout, 1).unsqueeze(1)
    if use_cuda:
        xx = xx.cuda()
    roll_pxs = torch.zeros(nrollout, rollout_len)
    with torch.no_grad():
        for idx in range(rollout_len):
            out = model(xx)
            smpl = torch.normal(out[:, 0], out[:, 1])
            roll_pxs[:, idx] = smpl
            xx = torch.cat((xx[..., 1:], smpl.unsqueeze(-1).unsqueeze(-1)), -1)
    return roll_pxs