#!/usr/bin/env python3

from scipy.io import loadmat 
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle


parser = ArgumentParser()
parser.add_argument("--inputSimMat", type=str)
parser.add_argument("--GT", type=str)
parser.add_argument("--outputPR", type=str)
args = parser.parse_args()

input_path = args.inputSimMat
GT_path = args.GT
output_path = args.outputPR 


def calculateMatches(DD):
    n, n = DD.shape
    vmin = 0.8
    vmax = 1.2
    seq_len = 10
    Rwindow = 10
    no_seq = 16

    move_min = int(vmin * seq_len)
    move_max = int(vmax * seq_len) 
    move = np.arange(move_min, move_max + 1)
    v = move.astype(float) / seq_len 

    if n < move_min + 2*Rwindow + no_seq + seq_len or n < seq_len:
        return None

    matches = np.nan * np.ones((n, 2))
    for N in range(move_min + 2*Rwindow + no_seq + seq_len -1, n):
        idx_add = np.tile(np.arange(0, seq_len), (len(v), 1))
        idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)

        idx_add1 = np.tile(idx_add[:,-1], (seq_len, 1)).T - idx_add

        # this is where our trajectory starts
        n_start = N + 1 - seq_len 
        if n_start < no_seq:
            continue

        x = np.tile(np.arange(n_start, n_start + seq_len), (len(v), 1))

        y_max = n_start - no_seq
        xx = x * n

        score = []
        final_idx = []
        flatDD = DD.flatten('F')
        for s in range(y_max+1 - move_min + 1):  # 0 ~ N+1-move_min
            y = np.copy(idx_add + s)
            y1 = np.copy(idx_add1 + s)
            y[y > y_max] = y_max
            y1[y1 > y_max] = y_max
            idx = (xx + y).astype(int)
            idx1 = (xx + y1).astype(int)
            ds = np.sum(flatDD[idx], 1)
            ds1 = np.sum(flatDD[idx1], 1)
            tmp = np.argmin(ds)
            tmp1 = np.argmin(ds1)
            if tmp <= tmp1:
                score.append(ds[tmp])
                final_idx.append(min(n-1, s + int(idx_add[tmp, -1])))
            else:
                score.append(ds1[tmp1])
                final_idx.append(s)

        score = np.array(score)
        min_idx = np.argmin(np.array(score))
        min_value = score[min_idx]
        window = np.arange(np.max((0, min_idx - Rwindow // 2)),
                           np.min((len(score), min_idx + Rwindow // 2)))
        not_window = list(set(range(len(score))).symmetric_difference(set(window)))  # xor
        if len(not_window) < Rwindow:
            continue
        min_value_2nd = np.min(score[not_window])
        match = [final_idx[min_idx], min_value / min_value_2nd]
        matches[N, :] = match

    return matches



dist_mat = 1 - np.loadtxt(input_path)

matches = calculateMatches(dist_mat)

# draw PR curve
row = matches.shape[0]
groundtruthMat = loadmat(GT_path)
groundtruthMat = groundtruthMat['truth'][::2,::2]
gt_loop = np.count_nonzero(np.sum(groundtruthMat, 1))
pr = []
if len(matches) > 0: 
    for mu in np.arange(0, 1, 0.01):
        # The LARGER the score, the WEAKER the match.
        idx = np.copy(matches[:, 0])
        # remove the weakest matches
        idx[matches[:, 1] > mu] = np.nan

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
