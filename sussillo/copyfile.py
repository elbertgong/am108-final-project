import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np

# embedding, RNN, vocab layer
# don't even need embedding for now
# self.embedding = nn.Embedding(3,hidden_dim)

class ThreeBitRNN(nn.Module):
    def __init__(self, hidden_size=100, seq_len=20):
        super(ThreeBitRNN, self).__init__()
        self.seq_len = seq_len
        self.hidden = Variable(torch.zeros(1,1,hidden_size))
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
    def get_hidden(self):
        return self.hidden
    def set_hidden(self, h):
        self.hidden = Variable(h.data)
    def forward(self, x):
        out = x.unsqueeze(1)
        h = self.get_hidden()
        out,h = self.rnn(out,h) # i think out[-1] is h
        self.set_hidden(h)
        out = out.squeeze(1)
        out = self.output_layer(out)
        return out
    def all_hiddens(self, x):
        out = x.unsqueeze(1)
        h = self.get_hidden()
        out,h = self.rnn(out,h) # i think out[-1] is h
        out = out.squeeze(1)
        return out


import numpy as np
import random
# how many data points
def genxy(timesteps, p = 0.5):
    # params for binomial distribution; n=1 for binary change / no change, 
    # p is the odds of a change at any given time step
    n = 1

    # number of channels for input / output, the dimensionality of the data set.
    channels = 3

    # 2 bits per channel, corresponding to +1/-1
    choices = [-1,1]

    # timeseries_seed dictates whether or not a bit will flip at any given time step.
    timeseries_seed = np.random.binomial(n,p,timesteps)

    # X is the input data
    X = np.zeros([channels,timeseries_seed.size])

    for i in range(len(timeseries_seed)):
        if timeseries_seed[i]==1:
            # if we change a bit, choose which channel it changes on
            channel = np.random.choice(range(channels))
            # then choose what to change the bit to, -1 or +1
            X[channel,i] = np.random.choice(choices)
            
    # y is data labels, system state computed from X series
    y = np.zeros_like(X)

    # state keeps track of what bits should be -1 or 1 given X(t)
    state = -np.ones([channels])
    for i in range(len(timeseries_seed)):
        if sum(abs(X[:,i]))>0:
            # if there is an input at X(t), we update the state vector to reflect it's new value
            state[np.where(abs(X[:,i]))] = X[np.where(abs(X[:,i])),i]
        # write the state to y(t)
        y[:,i] = state

    return X, y

# # lets take a look
# print("first 10 X\n",X[:,:10])
# print("first 10 y\n",y[:,:10])
