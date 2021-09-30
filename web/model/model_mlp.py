import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, n_input, hidden_size1, hidden_size2, hidden_size3, n_output):
        super(MLP,self).__init__()
        self.hidden1 = nn.Linear(n_input, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2, hidden_size3)
        self.predict = nn.Linear(hidden_size3, n_output)
    def forward(self,input):
        out = self.hidden1(input)
        # m1 = nn.BatchNorm1d(hidden1)
        # out = m1(out)
        out = torch.relu(out)
        out = self.hidden2(out)
        # m2 = nn.BatchNorm1d(hidden2)
        # out = m2(out)
        out = torch.relu(out)
        out = self.hidden3(out)
        # m3 = nn.BatchNorm1d(hidden3)
        # out = m3(out)
        out = torch.tanh(out)
        out =self.predict(out)

        return out
