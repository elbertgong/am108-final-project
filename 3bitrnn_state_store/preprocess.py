import numpy as np
import pandas as pd
import h5py

# Code snippets to process 3bit data for feeding into LSTMviz

# I'm also not sure I'm saving the arrays into hdf5 with correct dimensionality

hids = pd.read_csv('../rnndata/state_store.csv',header=None).values
f = h5py.File('states.hdf5','w')
# f['states'] = hids
f.create_dataset('states', data=hids)

traj = pd.read_csv('../rnndata/state_store_inputs.csv',header=None).values
word_dict = {'[0.0.0.]':1, '[0.0.1.]':2, '[0.0.-1.]':3, '[0.1.0.]':4, '[0.-1.0.]':5,
        '[1.0.0.]':6, '[-1.0.0.]':7}

train_words = []
for i in range(len(traj)):
    train_words.append(word_dict[np.array_str(traj[i]).replace(' ','')])

train_words = np.array(train_words)
f = h5py.File('train.hdf5','w')
f.create_dataset('word_ids',data=train_words)

def write(d,outfile):
    out = open(outfile, "w")
    items = [(v,k) for k,v in d.items()]
    items.sort()
    for v, k in items: # what
        print (k, v, file=out)
    out.close()

write(word_dict,'words.dict')

# How to open an hdf5 file
f = h5py.File('train.hdf5', 'r')
# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
# Get the data
data = list(f[a_group_key])
