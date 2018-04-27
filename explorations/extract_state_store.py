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
hids = np.zeros((140,HIDDEN_SIZE))
traj = np.zeros((140,3))

traj[20,2]=1
traj[40,1]=1
traj[60,0]=1
traj[80,2]=-1
traj[100,1]=-1
traj[120,0]=-1


#traj[20,2]=1
#traj[40,1]=1
#traj[60,0]=1
#traj[80,2]=-1
#traj[100,1]=-1
#traj[120,0]=-1
#traj[140,1]=1
#traj[160,2]=1
#traj[180,1]=-1
#traj[200,0]=1
#traj[220,2]=-1
#traj[240,1]=1
#traj[260,0]=-1
#traj[280,2]=1
#traj[300,0]=1
#traj[320,1]=-1
#traj[340,2]=-1
#traj[360,0]=-1
#traj[380,1]=1
#traj[400,2]=1
#traj[420,1]=-1
#traj[440,1]=1
#traj[460,0]=1
#traj[480,2]=-1
#traj[500,2]=1
#traj[520,1]=-1
#traj[540,0]=-1
#traj[560,1]=1
#traj[580,2]=-1
#traj[600,0]=1
#traj[620,1]=-1
#traj[640,1]=1
#traj[660,0]=1
#traj[680,2]=-1
#traj[700,1]=-1
#traj[720,0]=-1
#traj[740,1]=1
#traj[760,2]=1
#traj[780,1]=-1
#traj[800,0]=1
#traj[820,2]=-1
#traj[840,1]=1
#traj[860,0]=-1
#traj[880,2]=1
#traj[900,0]=1
#traj[920,1]=-1
#traj[940,2]=-1
#traj[960,0]=-1
#traj[980,1]=1
#traj[1000,2]=1
#traj[1020,1]=-1
#traj[1040,1]=1
#traj[1060,0]=1
#traj[1080,2]=-1
#traj[1100,2]=1
#traj[1120,1]=-1
#traj[1140,0]=-1
#traj[1160,1]=1
#traj[1180,2]=-1
#traj[1200,0]=1
#traj[1220,1]=-1
#traj[1040,1]=1
#traj[1060,0]=1
#traj[1080,2]=-1
#traj[1100,2]=1
#traj[1120,1]=-1
#traj[1140,0]=-1
#traj[1160,1]=1
#traj[1180,2]=-1
#traj[1200,0]=1
#traj[1220,1]=-1
#traj[1240,1]=1
#traj[1260,0]=1
#traj[1280,2]=-1
#traj[1300,2]=1
#traj[1320,1]=-1
#traj[1340,0]=-1
#traj[1360,1]=1
#traj[1380,2]=-1
#traj[1400,0]=1
#traj[1420,1]=-1
#traj[1440,0]=-1
#traj[1460,1]=1
#traj[1480,2]=-1
#traj[1500,0]=1
#traj[1520,1]=-1
#traj[1540,1]=1
#traj[1560,0]=1
#traj[1580,2]=-1
#traj[1600,2]=1
#traj[1620,1]=-1
#traj[1640,0]=-1
#traj[1660,1]=1
#traj[1680,2]=-1
#traj[1700,0]=1
#traj[1720,1]=-1
#traj[1740,1]=1
#traj[1760,0]=1
#traj[1780,2]=-1
#traj[1800,2]=1
#traj[1820,1]=-1
#traj[1840,0]=-1
#traj[1860,1]=1
#traj[1880,2]=-1
#traj[1900,0]=1
#traj[1920,1]=-1
#traj[1940,0]=-1
#traj[1960,1]=1
#traj[1980,2]=-1
#traj[2000,0]=1
#traj[2020,1]=-1
#traj[2040,1]=1
#traj[2060,0]=1
#traj[2080,2]=-1
#traj[2100,2]=1
#traj[2120,0]=-1
#traj[2140,1]=-1
#traj[2160,1]=1
#traj[2180,2]=-1
#traj[2200,0]=1
#traj[2220,1]=-1
#traj[2040,2]=1
#traj[2060,0]=1
#traj[2080,2]=-1
#traj[2100,1]=1
#traj[2120,0]=-1
#traj[2140,1]=-1
#traj[2160,1]=-1
#traj[2180,2]=1
#traj[2200,0]=1
#traj[2220,1]=-1
#traj[2240,2]=-1
#traj[2260,0]=1
#traj[2280,0]=-1
#traj[2300,2]=1
#traj[2320,1]=1
#traj[2340,0]=-1
#traj[2360,1]=-1
#traj[2380,2]=-1
#traj[2400,0]=1
#traj[2420,1]=1

np.savetxt("rnndata/state_store_inputs.csv", traj, delimiter=',')
for i in range(n): # condition averaging
    trajvar = Variable(torch.Tensor(traj), requires_grad=False)
    model.set_hidden(Variable(torch.zeros(1,1,HIDDEN_SIZE)))
    hids += model.all_hiddens(trajvar).data.numpy()
hids /= n
np.savetxt("rnndata/state_store.csv", hids, delimiter=",")

# go from inputs to target outputs on 0-7 scale
traj = pd.read_csv('rnndata/state_store_inputs.csv',header=None).values
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
