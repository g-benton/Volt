import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable  


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
    def __init__(self, train_x, train_y, seq_len, hidden_size, 
                 num_layers, batch_size=128):
        super(LSTM, self).__init__()
        
        self.train_x = train_x
        self.train_y = train_y
        
        self.norm_y = (train_y - train_y.mean())/train_y.std()

        self.dset = SequenceDataset(self.norm_y, sequence_length=seq_len)
        self.trainloader = DataLoader(self.dset, batch_size=batch_size,
                                      shuffle=True)
        
        self.num_classes = 1
        self.num_layers = num_layers
        self.input_size = seq_len
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=seq_len, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, 2) #fully connected last layer

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
    
    def Loss(self, targets, outputs):
        dist = torch.distributions.Normal(outputs[:, 0], outputs[:, 1])
        return -dist.log_prob(targets).sum()

    def Train(self, epochs, display=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        num_batches = len(self.trainloader)
        total_loss = 0
        self.train()
        for epoch in range(epochs):
            for X, y in self.trainloader:
                X = X.to(self.train_x.device)
                y = y.to(self.train_x.device)
                output = self(X)
                loss = self.Loss(y, output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            if display:
                if epoch%50 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Train loss: {avg_loss}, Epoch: {epoch}")

    
    def Forecast(self, test_x, nsample=50):
        rollout_len = test_x.shape[0]
        xin, xout = self.dset[len(self.dset)-1]
        xx = torch.cat((xin[0, 1:], xout.unsqueeze(0)))
        xx = xx.repeat(nsample, 1).unsqueeze(1)
        xx = xx.to(self.train_x.device)
        roll_pxs = torch.zeros(nsample, rollout_len)
        with torch.no_grad():
            for idx in range(rollout_len):
                out = self(xx)
                smpl = torch.normal(out[:, 0], out[:, 1])
                roll_pxs[:, idx] = smpl
                xx = torch.cat((xx[..., 1:], smpl.unsqueeze(-1).unsqueeze(-1)), -1)
        return roll_pxs * self.train_y.std() + self.train_y.mean()
        