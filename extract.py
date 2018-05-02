import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import time
import argparse
import pandas as pd
from helpers import timeSince, asMinutes, reshape
from models import ThreeBitRNN
from gen_data import genxy

HIDDEN_SIZE = 100

# Generates hidden state csvs: no_inputs.csv, 0perturb.csv, 1perturb.csv, 2perturb.csv
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

# get RNN weight matrix
m = model.rnn.state_dict()['weight_hh_l0'].numpy()
np.save('rnndata/weight_hh_l0.npy', m)

# test a random trajectory
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

# create special world tour trajectory and extract hidden states (condition-averaged)
model = ThreeBitRNN(hidden_size=HIDDEN_SIZE)
model.load_state_dict(torch.load('rnndata/model.pkl'))
n = 101
hids = np.zeros((320,HIDDEN_SIZE))
traj = np.zeros((320,3))
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
np.savetxt("rnndata/worldtour_inputs.csv", traj, delimiter=',')
for i in range(n): # condition averaging
    trajvar = Variable(torch.Tensor(traj), requires_grad=False)
    model.set_hidden(Variable(torch.zeros(1,1,HIDDEN_SIZE)))
    hids += model.all_hiddens(trajvar).data.numpy()
hids /= n
np.savetxt("rnndata/worldtour.csv", hids, delimiter=",")

# go from inputs to target outputs on 0-7 scale
traj = pd.read_csv('rnndata/worldtour_inputs.csv',header=None).values
true_out = np.zeros((traj.shape[0],8))
true_out[0,0] = 1 # start with all the lights off
state = -np.ones([3])
for i in range(1,len(traj)):
    if sum(abs(traj[i]))>0:
        # if there is an input at X(t), we update the state vector to reflect its new value
        state[np.where(abs(traj[i]))] = traj[i,np.where(abs(traj[i]))]
        true_out[i,4*(state[0]==1)+2*(state[1]==1)+(state[2]==1)]=1
    else:
        true_out[i] = true_out[i-1]

# then compare target outputs to model's outputs
model = ThreeBitRNN(hidden_size=HIDDEN_SIZE)
model.load_state_dict(torch.load('rnndata/model.pkl'))
traj = Variable(torch.Tensor(traj), requires_grad=False)
out = model(traj)
out = F.softmax(out,dim=1)
out = out.data.numpy()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
gs = GridSpec(4, 2)
for i in range(8):
    ax = fig.add_subplot(gs[i%4, i//4])
    ax.plot(out[:,i])
    ax.plot(true_out[:,i])
    ax.set_ylim(-0.1,1.1)
    # ax0.set_ylabel('x')
    # ax0.set_xticks([])
    # ax0.set_xlim(0, 20)
plt.show()

