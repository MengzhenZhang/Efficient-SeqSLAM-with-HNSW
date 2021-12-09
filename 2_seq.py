
from pyseqslam.parameters import defaultParameters
from pyseqslam.utils import AttributeDict
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.io import loadmat
import os
import numpy as np
from pyseqslam.seqslam import *
import pickle

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--GT", type=str)
parser.add_argument("--outputPR", type=str)
args = parser.parse_args()

groundtruthPath = args.GT
params = defaultParameters()

# set the custom parameters
# only use the right camera
ds = AttributeDict()
ds.imagePath = args.dataset

num_of_items = len(os.listdir(ds.imagePath))
ds.extension = '.jpg'
ds.dataFormat = '04'  # 00001
ds.imageSkip = 2  # use every n-nth image
ds.imageIndices = range(1, num_of_items+1, ds.imageSkip)
ds.preprocessing = AttributeDict()
# ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
ds.crop = []
params.dataset = [ds, deepcopy(ds)]

# now process the dataset
seq = SeqSLAM(params)
results = seq.findLoopClosure()

# draw PR curve
row = results.matches.shape[0]
groundtruthMat = loadmat(groundtruthPath)
groundtruthMat = groundtruthMat['truth'][::2,::2]
gt_loop = np.count_nonzero(np.sum(groundtruthMat, 1))
pr = []
if len(results.matches) > 0: 
    for mu in np.arange(0, 1, 0.01):
        # The LARGER the score, the WEAKER the match.
        idx = np.copy(results.matches[:, 0])
        # remove the weakest matches
        idx[results.matches[:, 1] > mu] = np.nan

        loopMat = np.zeros((row, row))
        for i in range(row):
            if not np.isnan(idx[i]):
                loopMat[i, int(idx[i])] = 1

        p_loop = np.sum(loopMat)
        if p_loop != 0:
            TP = np.sum(loopMat * groundtruthMat)
            pre = TP / p_loop
            rec = TP / gt_loop
            pr.append([pre, rec])

pr = np.array(pr)
PR_curve_file = args.outputPR
with open(PR_curve_file, 'wb') as f:
    pickle.dump(pr, f)
_, ax = plt.subplots()
ax.plot(pr[:, 1], pr[:, 0], '-o')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.axis([0, 1.05, 0, 1.05])
ax.grid()
plt.show()
