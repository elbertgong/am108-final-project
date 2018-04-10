import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# embedding, RNN, vocab layer
# don't even need embedding for now
# self.embedding = nn.Embedding(3,hidden_dim)

class ThreeBitRNN(nn.Module):
    def __init__(self, hidden_size=100, seq_len=20):
        super(ThreeBitRNN, self).__init__()
        self.seq_len = seq_len
        self.hidden = Variable(torch.zeros(1,1,3))
        self.rnn = nn.RNN(input_size=3,hidden_size=hidden_size)
        # other relevant args: num_layers, nonlinearity, dropout
        # rnn input shape: seq_len,bs,input_size
        # hidden shape: num_layers*num_directions,bs,hidden_size
        # output shape: seq_len,bs,hidden_size*num_directions
        self.output_layer = nn.Sequential(OrderedDict([
            ('h2o',nn.Linear(hidden_size, 8)),
            # ('tanh',nn.Tanh()),
            # ('drp',nn.Dropout(vocab_layer_dropout)),
            # ('lsft',nn.LogSoftmax(dim=-1))
        ]))
    def init_hidden(self):
        return self.hidden
    def save_hidden(self, h):
        self.hidden = Variable(h.data)
    def forward(self, x):
        out = x.unsqueeze(1)
        h = self.get_hidden()
        out,h = self.rnn(x,h) # i think out[-1] is h
        self.set_hidden(h)
        out = out.squeeze(1)
        out = self.output_layer(out)
        return out