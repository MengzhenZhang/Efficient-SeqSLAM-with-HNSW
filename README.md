# Efficient-SeqSLAM-with-HNSW
This repo shares code used in our manuscript

Note that I did not include datasets in this repo. They are publicly available here `https://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/data.html`.

Please change the path to datasets according to your case.


**Main step to reproduce our results**

1. Run `generateDict.py` to generate a visual dictionary offline.

2. To avoid converting images to VLADs in each experiment, we can save the VLADs in a pickle file.
Run `generateVLADs.py` to generate  VLADs.

3. Run `visualization.py` to reproduce our results including correct and incorrect pairs of matches, confusion matrix.

4. Run `differentLengthOfSeq.py` to reproduce the result about different lengths of sequence.
