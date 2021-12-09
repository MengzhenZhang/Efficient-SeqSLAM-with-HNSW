import cv2
import numpy as np
import glob
import itertools
import pickle
from sklearn.cluster import KMeans
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--outputDict', type=str)
args = parser.parse_args()


def describeORB(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


path = args.dataset
#path = 'datasets/NewCollege/Images'

descriptors = list()

for imagePath in sorted(glob.glob(path+"/*.jpg")):
    print(imagePath)
    im = cv2.imread(imagePath)
    kp, des = describeORB(im)
    descriptors.append(des)


# flatten list
descriptors = list(itertools.chain.from_iterable(descriptors))
# list to array
descriptors = np.asarray(descriptors)


k = 64
visualDictionary = KMeans(
    n_clusters=k, init='k-means++', tol=0.0001).fit(descriptors)

dictFile = args.outputDict
#dictFile = "dictionaryFiles/dictFromNewCollege.pickle"

with open(dictFile, 'wb') as f:
    pickle.dump(visualDictionary, f)
