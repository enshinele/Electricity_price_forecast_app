import torch
import torch.nn as nn
class Seq2Seq(nn.Module):
    def __init__(self, input_size, n_hidden, num_layers, n_class):
        super(Seq2Seq, self).__init__()
        self.num_features = input_size
        self.num_layers = num_layers
        self.hidden_size = n_hidden
        self.encoder = nn.RNN(input_size=input_size, hidden_size=n_hidden, dropout=0.5) # encoder
        self.decoder = nn.RNN(input_size=input_size, hidden_size=n_hidden, dropout=0.5) # decoder
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = x.reshape(len(x), -1, self.num_features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x = x.transpose(0, 1)
        c0 = torch.zeros_like(x)

        # h_t : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, h_t = self.encoder(x, h0)
        # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.decoder(x, h_t)

        out = self.fc(outputs[-1, :, :]) # model : [n_step+1, batch_size, n_class]
        return out

