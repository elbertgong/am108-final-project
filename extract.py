import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import time
import argparse
from helpers import timeSince, asMinutes
from models import ThreeBitRNN
from gen_data import genxy

traj = Variable(torch.Tensor(torch.zeros((40,3))), requires_grad=False)
model = ThreeBitRNN(hidden_size=100)
model.load_state_dict(torch.load('model.pkl'))
no_inputs = model.all_hiddens(traj).data.numpy()
np.savetxt("no_inputs.csv", no_inputs, delimiter=",")
for i in range(3):
    traj = Variable(torch.Tensor(torch.zeros((40,3))), requires_grad=False)
    traj[20,i] = 1
    out = model.all_hiddens(traj).data.numpy()
    np.savetxt(str(i)+"perturb.csv", out, delimiter=",")

# get weight matrix
m = model.rnn.state_dict()['weight_hh_l0'].numpy()
np.save('weight_hh_l0.npy', m)