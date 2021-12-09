
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
parser.add_argument('--vladFile', type=str)
parser.add_argument('--GT', type=str)
parser.add_argument('--outputTime', type=str)
parser.add_argument('--outputPR', type=str)
args = parser.parse_args()

### 1. Load VLADs
vladFile = args.vladFile
#vladFile = "vladFiles/NewCollegeVLADs.pickle"

with open(vladFile, 'rb') as f:
    VLADs = pickle.load(f)

### 2. Construct HNSW incrementally and search at the same time
dim = 10 * len(VLADs[0])
num_elements = len(VLADs)

hnsw = HNSW('l2', m = 64, m0=64, ef=100)
#p = hnswlib.Index(space='l2', dim=dim)
# p.init_index(max_elements=num_elements, ef_construction=100, M=64)
# p.set_ef(100)

image_num = len(VLADs)
matches = np.nan * np.ones([image_num, 2])

timing = []

for index, vlad in enumerate(VLADs):   
    print('index: ', index)
    
    # construction start from index=19 
    if index >= 19:
        
        # add node 10 frames earlier
        vlad_seq_add = VLADs[index-19:index-9]
        node_add = np.array(list(itertools.chain.from_iterable(vlad_seq_add)), dtype='float32')
        hnsw.add(node_add)
        #p.add_items(node_add, index-10)
 
        # detection start from index=100
        if index >= 100:     
            
            vlad_seq = VLADs[index-9:index+1]
            node = np.array(list(itertools.chain.from_iterable(vlad_seq)), dtype='float32')
            
            # start timing
            t1 = time.time()
            
            indice, distance = hnsw.search(node, 1)[0]
            indice = indice + 9

            # end timing
            t2 = time.time()
            timing.append([index, t2-t1])

            matches[index, 0] = int(indice)
            matches[index, 1] = distance   


matches[:,1] = matches[:,1] / np.nanmax(matches[:,1])

### 3. Search time 
timing_file = args.outputTime
# timing_file = "timeFiles/NewCollegeHNSW.pickle"

with open(timing_file, 'wb') as f:
    pickle.dump(np.array(timing), f)

timing_array = np.array(timing)
plt.plot(timing_array[:,0], timing_array[:,1], 'r-')
plt.xlabel('Number of Images')
plt.ylabel('Search Time')
plt.show()

### 4. PR curve
# load GT matrix
groundtruthPath = args.GT
#groundtruthPath = 'datasets/NewCollege/NewCollegeGroundTruth.mat'
groundtruthMat = loadmat(groundtruthPath)

# extract right view
groundtruthMat = groundtruthMat['truth'][::2,::2]

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
        print("mu with detected: ", mu)
        print("TP: ", TP)
        print("p_loop: ", p_loop)
        print("gt_loop: ", gt_loop)

pr = np.array(pr)

fig, ax = plt.subplots()
ax.plot(pr[:, 1], pr[:, 0], '-o')
print("recall: ", pr[:,1])
print("precision: ", pr[:,0])
ax.set_title('PR Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid()
plt.axis([0, 1.05, 0, 1.05])
plt.show()

PR_curve_file = args.outputPR
# PR_curve_file = "PrecisionRecallFiles/PrNewCollege.pickle'

with open(PR_curve_file, 'wb') as f:
    pickle.dump(pr, f)

