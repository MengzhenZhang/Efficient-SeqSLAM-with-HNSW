
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
parser.add_argument('--seq', type=str)
parser.add_argument('--seq_bow', type=str)
parser.add_argument('--seq_vlad', type=str)
parser.add_argument('--seq_vlad_hnsw', type=str)
args = parser.parse_args()

### 1. Load all PR files
prFile1 = args.seq
prFile2 = args.seq_bow
prFile3 = args.seq_vlad
prFile4 = args.seq_vlad_hnsw

with open(prFile1, 'rb') as f:
    pr1 = pickle.load(f)

with open(prFile2, 'rb') as f:
    pr2 = pickle.load(f)

with open(prFile3, 'rb') as f:
    pr3 = pickle.load(f)

with open(prFile4, 'rb') as f:
    pr4 = pickle.load(f)


fig, ax = plt.subplots()
ax.plot(pr1[:, 1], pr1[:, 0], '-', label="seq")
ax.plot(pr2[:, 1], pr2[:, 0], '--o', label="seq+bow")
ax.plot(pr3[:, 1], pr3[:, 0], '-.', label="seq+vlad")
ax.plot(pr4[:, 1], pr4[:, 0], ':', label="seq+vlad+hnsw")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid()
ax.legend(loc="lower left")
plt.axis([0, 1, 0, 1.05])
plt.show()



