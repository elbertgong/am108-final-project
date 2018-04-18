import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import time
import argparse
from helpers import timeSince, asMinutes, reshape
from models import ThreeBitRNN
from gen_data import genxy

traj = Variable(torch.Tensor(torch.zeros((40,3))), requires_grad=False)
model = ThreeBitRNN(hidden_size=100)
model.load_state_dict(torch.load('rnndata/model.pkl'))
no_inputs = model.all_hiddens(traj).data.numpy()
np.savetxt("rnndata/no_inputs.csv", no_inputs, delimiter=",")
for i in range(3):
    traj = Variable(torch.Tensor(torch.zeros((40,3))), requires_grad=False)
    traj[20,i] = 1
    out = model.all_hiddens(traj).data.numpy()
    np.savetxt("rnndata/perturb"+str(i)+".csv", out, delimiter=",")

# get weight matrix
m = model.rnn.state_dict()['weight_hh_l0'].numpy()
np.save('rnndata/weight_hh_l0.npy', m)

# test a random traj
criterion = nn.CrossEntropyLoss()
model = ThreeBitRNN(hidden_size=100)
model.load_state_dict(torch.load('rnndata/model.pkl'))
for _ in range(10):
    inn, out = reshape(genxy(100,0.25))
    inn = Variable(torch.Tensor(inn), requires_grad=False)
    out = Variable(torch.LongTensor(out))
    outt = model(inn)
    _, preds = torch.max(outt,1)
    print(criterion(outt,out).data[0])
    print(sum(preds==out).data[0])

# special traj
model = ThreeBitRNN(hidden_size=100)
model.load_state_dict(torch.load('rnndata/model.pkl'))
n = 101
hids = np.zeros((80,100))
for i in range(n):
    traj = Variable(torch.Tensor(torch.zeros((80,3))), requires_grad=False)
    traj[20,0]=1
    traj[40,1]=1
    traj[60,2]=1
    hids += model.all_hiddens(traj).data.numpy()

hids /= n
np.savetxt("rnndata/3flips.csv", hids, delimiter=",")