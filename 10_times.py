
import numpy as np
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
#import hnswlib
from hnsw import HNSW
import itertools

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seq_vlad', type=str)
parser.add_argument('--seq_vlad_hnsw', type=str)
args = parser.parse_args()

### 1. Load all PR files
timeFile1 = args.seq_vlad
timeFile2 = args.seq_vlad_hnsw

with open(timeFile1, 'rb') as f:
    time1 = pickle.load(f)
    time1 = np.array(time1)

with open(timeFile2, 'rb') as f:
    time2 = pickle.load(f)



fig, ax = plt.subplots()

ax.plot(time1[:, 0], time1[:, 1], '-.', label="seq+vlad")
ax.plot(time2[:, 0], time2[:, 1], ':', label="seq+vlad+hnsw")
ax.set_xlabel('Number of Images')
ax.set_ylabel('Time (s)')
ax.grid()
ax.legend(loc="upper left")
plt.show()



