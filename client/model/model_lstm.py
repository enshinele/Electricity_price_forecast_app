import torch
from torch import nn
# LSTM
# __init__ is basically a function which will "initialize"/"activate" the properties of the class for a specific object
# self represents that object which will inherit those properties
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_features = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = x.reshape(len(x), -1, self.num_features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        # Select the output at the last moment
        out = self.fc(out[:, -1, :])
        return out

