import cv2
import numpy as np
import glob
import pickle
from numpy.linalg import norm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--dict', type=str)
parser.add_argument('--outputVLAD', type=str)
args = parser.parse_args()


def describeORB(image):
    orb=cv2.ORB_create()
    kp, des=orb.detectAndCompute(image,None)
    return kp,des


path = args.dataset
#path = 'datasets/NewCollege/Images'  

dictFile = args.dict
#dict_file = "dictionaryFiles/dictFromNewCollege.pickle"   # To keep unrelated, NewCollege should use the dictionary generated from CityCentre, and vice verse.

with open(dictFile, 'rb') as f:
    visualDictionary = pickle.load(f)


def VLAD(X, visualDictionary):

    # number of clusters: 64
    k = visualDictionary.n_clusters

    # 64 center vectors
    centers = visualDictionary.cluster_centers_

    # predicted labels for the query descriptor set X
    predictedLabels = visualDictionary.predict(X)

    # 500 * 32 = 500 desc for each image  *  length of each ORB desc
    _, d = X.shape

    # init VLAD by zeros.
    V = np.zeros([k, d])

    # for all the clusters
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if i in predictedLabels:
            # add the residuals
            V[i] = np.sum(X[predictedLabels == i, :]-centers[i], axis=0)

    # flatten the VLAD matrix to a vector
    V = V.flatten()

    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V, V))

    # return VLAD as a vector
    return V

    
def distance(seq1, seq2):
    dist = np.array(seq1) - np.array(seq2)
    dist_norm = norm(dist, axis=1)
    dist_sum = sum(dist_norm)
    return dist_sum


# only consider images on one side
imagePaths = sorted(glob.glob(path+"/*.jpg"))[::2]

image_num = len(imagePaths)
matches = np.nan * np.ones([image_num, 2])

vlad_history = []

for index, imagePath in enumerate(imagePaths):
    print('image: ', imagePath)
    print('index: ', index)
    im = cv2.imread(imagePath)
    kp, des = describeORB(im)

    # for each image, compute VLAD
    vlad = VLAD(des, visualDictionary)

    # add current vlad to hisroty seq
    vlad_history.append(vlad)

    # detection start from 100 frame
    if index >= 100:

        vlad_seq_query = vlad_history[index-9:index+1]
        min_dist = np.inf
        min_index = None


        # only search (9, index-10)
        for i in range(9, index-10):
            vlad_seq_search = vlad_history[i-9:i+1]
            dist = distance(vlad_seq_query, vlad_seq_search)
            if dist < min_dist:
                min_dist = dist
                min_index = i

        matches[index, 0] = min_index
        matches[index, 1] = min_dist


vladFile = args.outputVLAD
#vlad_file = "vladFiles/NewCollegeVLADs.pickle"

with open(vladFile, 'wb') as f:
    pickle.dump(vlad_history, f)

