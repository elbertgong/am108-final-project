#!/usr/bin/python

import sys
import numpy as np
import scipy.io

path = sys.argv[1]
dest = path.replace('.npy','')

data = np.load(path)

scipy.io.savemat(dest, dict(data=data))
