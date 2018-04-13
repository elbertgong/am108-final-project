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
