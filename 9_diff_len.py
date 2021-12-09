
import numpy as np
import pickle
import matplotlib.pyplot as plt
import hnswlib
import itertools
from scipy.io import loadmat
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--vlad", type=str)
parser.add_argument("--GT", type=str)
args = parser.parse_args()


# load vlad files
vlads_file = args.vlad

with open(vlads_file, 'rb') as f:
    VLADs = pickle.load(f)


# load GT matrix
groundtruthPath = args.GT
groundtruthMat = loadmat(groundtruthPath)
groundtruthMat

# extract right view
groundtruthMat = groundtruthMat['truth'][::2, ::2]
groundtruthMat

gt_loop = np.count_nonzero(np.sum(groundtruthMat, 1))

prs = []

#for ds in range(5, 51, 15):
for ds in [2,5,10,25,50]:
    dim = ds * len(VLADs[0])
    num_elements = len(VLADs)
    
    p = hnswlib.Index(space='l2', dim=dim)
    
    p.init_index(max_elements=num_elements, ef_construction=100, M=64)
    p.set_ef(100)
    
    
    image_num = len(VLADs)
    matches = np.nan * np.ones([image_num, 2])
    
    for index, vlad in enumerate(VLADs):
        print('index: ', index)
    
        # construction start from index= ds+9
        if index >= ds+9:
    
            # add node 10 frames earlier
            vlad_seq_add = VLADs[index-ds-9:index-9]
            node_add = np.array(
                list(itertools.chain.from_iterable(vlad_seq_add)), dtype='float32')
            p.add_items(node_add, index-10)
    
            # detection start from index=100
            if index >= 100:
    
                vlad_seq = VLADs[index-ds+1:index+1]
                node = np.array(
                    list(itertools.chain.from_iterable(vlad_seq)), dtype='float32')
    
                indice, distance = p.knn_query(node, k=1)
    
                matches[index, 0] = int(indice)
                matches[index, 1] = int(distance)
    
    
    matches[:, 1] = matches[:, 1] / np.nanmax(matches[:, 1])
    
    # Evaluation the matches by PR curve
    
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
            print("mu: ", mu, "pre: ", pre, " rec: ", rec)
            pr.append([pre, rec])
    
    pr = np.array(pr)

    prs.append(pr) 
    


# plt.subplots() is recommended by matplotlib
fig, ax = plt.subplots()
for pr in prs:
    ax.plot(pr[:, 1], pr[:, 0], '-o')

ax.set_title('PR Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid()
plt.axis([0, 1.05, 0, 1.05])
plt.legend(['p=2', 'p=5', 'p=10', 'p=25', 'p=50'], loc="lower left")
plt.show()