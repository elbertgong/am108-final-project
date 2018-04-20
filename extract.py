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

HIDDEN_SIZE = 100

traj = Variable(torch.Tensor(torch.zeros((40,3))), requires_grad=False)
model = ThreeBitRNN(hidden_size=HIDDEN_SIZE)
model.load_state_dict(torch.load('rnndata/model.pkl'))
model.set_hidden(Variable(torch.zeros(1,1,HIDDEN_SIZE)))
no_inputs = model.all_hiddens(traj).data.numpy()
np.savetxt("rnndata/no_inputs.csv", no_inputs, delimiter=",")
for i in range(3):
    traj = Variable(torch.Tensor(torch.zeros((40,3))), requires_grad=False)
    traj[20,i] = 1
    model.set_hidden(Variable(torch.zeros(1,1,HIDDEN_SIZE))) # doing this every time now
    out = model.all_hiddens(traj).data.numpy()
    np.savetxt("rnndata/"+str(i)+"perturb.csv", out, delimiter=",")

# get weight matrix
m = model.rnn.state_dict()['weight_hh_l0'].numpy()
np.save('rnndata/weight_hh_l0.npy', m)

# test a random traj
criterion = nn.CrossEntropyLoss()
model = ThreeBitRNN(hidden_size=HIDDEN_SIZE)
model.load_state_dict(torch.load('rnndata/model.pkl'))
for _ in range(10):
    inn, out = reshape(genxy(102,0.25))
    inn = Variable(torch.Tensor(inn), requires_grad=False)
    out = Variable(torch.LongTensor(out))
    model.set_hidden(Variable(torch.zeros(1,1,HIDDEN_SIZE)))
    outt = model(inn)
    _, preds = torch.max(outt,1)
    print(criterion(outt,out).data[0])
    print(sum(preds==out).data[0])

# special traj
model = ThreeBitRNN(hidden_size=HIDDEN_SIZE)
model.load_state_dict(torch.load('rnndata/model.pkl'))
n = 101
hids = np.zeros((320,HIDDEN_SIZE))
for i in range(n): # condition averaging
    traj = Variable(torch.Tensor(torch.zeros((320,3))), requires_grad=False)
    traj[20,0]=1
    traj[40,1]=1
    traj[60,0]=-1
    traj[80,1]=-1
    traj[100,2]=1
    traj[120,1]=1
    traj[140,0]=1
    traj[160,1]=-1
    traj[180,2]=-1
    traj[200,1]=1
    traj[220,2]=1
    traj[240,0]=-1
    traj[260,1]=-1
    traj[280,2]=-1
    traj[300,2]=-1 
    model.set_hidden(Variable(torch.zeros(1,1,HIDDEN_SIZE)))
    hids += model.all_hiddens(traj).data.numpy()

hids /= n
np.savetxt("rnndata/3flips.csv", hids, delimiter=",")