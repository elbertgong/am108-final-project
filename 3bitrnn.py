import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchtext import data
# from torchtext import datasets
# from torchtext.vocab import Vectors
import csv
import time
import argparse
from helpers import timeSince, asMinutes
from models import ThreeBitRNN

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_file','-mf',type=str,default='model.pkl',help='Model save target.')
parser.add_argument('--hidden_size','-hs',type=int,default=100,help='Size of hidden layer in RNN')
parser.add_argument('--seq_len','-sl',type=int,default=40,help='Sequence length (batch size=1 for now)')
parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')

sl = args.seq_len

''' NOTES
Simplifying assumption: batch size is just 1
Each epoch is one big loop, so I only reset hiddens sparingly
TODO: put data generating fn here
TODO: reshape the output according to specifications
'''

def reshape(oldmat):
    new = []
    for row in oldmat:
        x = 4*(row[0]==1)+2*(row[1]==1)+(row[2]==1)
        new.append(x)
    return new

# not using utils.DataLoader
train_in = 4 # n by 3
train_out = reshape([[0,0,0]]) # n
val_in = 4
val_out = 4

model = ThreeBitRNN(hidden_size=args.hidden_size)
params = list(filter(lambda x: x.requires_grad, model.parameters()))
optimizer = optim.Adadelta(params, lr=learning_rate, rho=args.rho, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(num_epochs):
    model.train()
    model.set_hidden(Variable(torch.zeros(1,1,3)))
    ctr = 0
    for i in range(0,len(train_out),sl):
        inp_batch = Variable(torch.Tensor(train_in[i*sl:(i+1)*sl,:]))
        trg_batch = Variable(torch.LongTensor(train_out[i*sl:(i+1)*sl]))
        optimizer.zero_grad()
        outputs = model(inp_batch)
        loss = criterion(outputs, trg_batch)
        loss.backward()
        optimizer.step()
        ctr += 1
        if ctr % 100 == 0:
            timenow = timeSince(start)
            print('Epoch [%d/%d], Iter [%d/%d], Time: %s, Loss: %4f'
                %(epoch+1, num_epochs, ctr, len(train_out)//sl, timenow, loss.data[0]))

    model.eval()
    model.set_hidden(Variable(torch.zeros(1,1,3)))
    losses = []
    for i in range(0,len(val_out),sl):
        inp_batch = Variable(torch.Tensor(val_in[i*sl:(i+1)*sl,:]))
        trg_batch = Variable(torch.LongTensor(val_out[i*sl:(i+1)*sl]))      
        outputs = model(inp_batch)
        loss = criterion(outputs, trg_batch)
        losses.append(loss.data[0])
    epoch_loss = sum(losses)
    timenow = timeSince(start)
    print('Epoch [%d/%d], Time: %s, Val Loss: %4f'
                %(epoch+1, num_epochs, timenow, epoch_loss))

    torch.save(model.state_dict(), args.model_file)
    print("Model saved at", args.model_file)