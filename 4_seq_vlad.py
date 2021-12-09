import numpy as np
import pickle
from numpy.linalg import norm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--vladFile', type=str)
parser.add_argument('--GT', type=str)
parser.add_argument('--outputTime', type=str)
parser.add_argument('--outputPR', type=str)
args = parser.parse_args()

def distance(seq1, seq2):
    dist = np.array(seq1) - np.array(seq2)
    dist_norm = norm(dist, axis=1)
    dist_sum = sum(dist_norm)   
    return dist_sum

### 1. Load VLADs
vladFile = args.vladFile
#vladFile = "vladFiles/NewCollegeVLADs.pickle"

with open(vladFile, 'rb') as f:
    VLADs = pickle.load(f)

matches = np.nan * np.ones([len(VLADs), 2])

timing = []

for index in range(len(VLADs)):
    # detection start from 100 frame
    if index >= 100:     

        vlad_seq_query = VLADs[index-9:index+1]
        min_dist = np.inf
        min_index = None
        
        # start timing
        t1 = time.time()
        
        # only search (9, index-10)
        for i in range(9, index-10):
            vlad_seq_search = VLADs[i-9:i+1]
            dist = distance(vlad_seq_query, vlad_seq_search)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        # end timing
        t2 = time.time()
        timing.append([index, t2-t1])
        
        matches[index, 0] = min_index
        matches[index, 1] = min_dist
        
matches[:,1] = matches[:,1] / np.nanmax(matches[:,1])

### 4. Evaluation the matches by PR curve

# load GT matrix
groundtruthPath = args.GT
groundtruthMat = loadmat(groundtruthPath)
groundtruthMat

# extract right view
groundtruthMat = groundtruthMat['truth'][::2,::2]
groundtruthMat

gt_loop = np.count_nonzero(np.sum(groundtruthMat, 1))
pr = []
row = matches.shape[0]

for mu in np.arange(0, 1, 0.01):
    idx = np.copy(matches[:, 0])  # The LARGER the score, the WEAKER the match.
    idx[matches[:, 1] > mu] = np.nan  # remove the weakest matches

    loopMat = np.zeros((row, row))
    for i in range(row):
        if not np.isnan(idx[i]):
            loopMat[i, int(idx[i])] = 1

    p_loop = np.sum(loopMat)
    TP = np.sum(loopMat * groundtruthMat)
    if p_loop != 0:
        pre = TP / p_loop
        rec = TP / gt_loop
        pr.append([pre, rec])

pr = np.array(pr)

PR_curve_file = args.outputPR
with open(PR_curve_file, 'wb') as f:
    pickle.dump(pr, f)

timing_file = args.outputTime
with open(timing_file, 'wb') as f:
    pickle.dump(timing, f)

# plt.subplots() is recommended by matplotlib
fig, ax = plt.subplots()
ax.plot(pr[:, 1], pr[:, 0], '-o')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid()
plt.axis([0, 1.05, 0, 1.05])
plt.show()

