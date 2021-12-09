# Efficient-SeqSLAM-with-HNSW
This repo shares code used in our manuscript

Note that the datasets NewCollege and CityCentre are not included in this repo due to their large size. They are publicly available here `https://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/data.html`.

Please change the path to datasets according to your case.

**Platform**:  Ubuntu 18.04 + Python 3.8 

**Main steps to reproduce our results**

1. Generate a visual dictionary offline.

   ```bash
   python 0_dict.py  --dataset datasets/NewCollege/Images	--outputDict files_dict/nc_dict.pickle
   ```
   
2. To avoid converting images to VLADs in each experiment, we can save the VLADs in a pickle file, and directly load them in following experiments.
   ```bash
   python 1_vlad.py --dataset datasets/NewCollege/Images --dict files_dict/nc_dict.pickle --outputVLAD files_vlad/nc_vlad.pickle
   ```
   
3. Generate PR curve and calculate search time.

   - SeqSLAM

     ```bash
     python 2_seq.py --dataset datasets/NewCollege/Images --GT datasets/NewCollege/NewCollegeGroundTruth.mat --outputPR files_pr/nc_seq.pickle
     ```
   
   - SeqSLAM + DBoW
   
     We first use a C++ lib --- DBoW3, an improved version of DBoW2, to generate similarity matrix, then calculate PR from this matrix.
   
     ```bash
     # (1) Install DBoW3 library from this link: https://github.com/rmsalinas/DBow3
     # (2) Compile loop_closure.cpp file to create an executable file
     # (3) Run the follow command to generate similarity matrix
     ./build/loop_closure orbvoc.dbow3 ../datasets/NewCollege/Images similarityMat.txt
     # (4) Run the following script which calculates PR from the above similarity matrix
     python convert_to_mat.py --inputSimMat similarityMat.txt --GT ../datasets/NewCollege/NewCollegeGroundTruth.mat  --outputPR ../files_pr/nc_seq_bow.pickle
     ```
   
   - SeqSLAM + VLAD
   
     Generate and save both PR and search time
   
     ```bash
     python 4_seq_vlad.py  --vladFile files_vlad/nc_vlad.pickle  --GT datasets/NewCollege/NewCollegeGroundTruth.mat --outputTime files_time/nc_seq_vlad.pickle --outputPR files_pr/nc_seq_vlad.pickle
     ```
   
   - SeqSLAM + VLAD + HNSW
   
     Generate and save both PR and search time
   
     ```bash
     python 5_seq_vlad_hnsw.py  --vladFile files_vlad/nc_vlad.pickle --GT datasets/NewCollege/NewCollegeGroundTruth.mat --outputTime files_time/nc_seq_vlad_hnsw.pickle --outputPR files_pr/nc_seq_vlad_hnsw.pickle
     ```
   
4. Show all PR curves

   ```bash
   python 6_pr.py --seq files_pr/nc_seq.pickle --seq_bow files_pr/nc_seq_bow.pickle --seq_vlad files_pr/nc_seq_vlad.pickle --seq_vlad_hnsw files_pr/nc_seq_vlad_hnsw.pickle
   ```

5. Show confusion matrix

   ```bash
   python 7_conf_mat.py --vladFiles files_vlad/nc_vlad.pickle --GT datasets/NewCollege/NewCollegeGroundTruth.mat 
   ```

6. Show correct and incorrect matches

   ```bash
   python 8_pairs.py --vlad files_vlad/nc_vlad.pickle --GT datasets/NewCollege/NewCollegeGroundTruth.mat --dataset datasets/NewCollege/Images
   ```

7. Show PRs for different length of sequence

   ```bash
   python 9_diff_len.py --vlad files_vlad/nc_vlad.pickle --GT datasets/NewCollege/NewCollegeGroundTruth.mat 
   ```

8. Show time comparison between SeqSLAM + VLAD and SeqSLAM + VLAD + HNSW

   ```bash
   python 10_compareTime.py --seq_vlad timeFiles/NewCollegeSeqVLAD.pickle --seq_vlad_hnsw timeFiles/NewCollegeHNSW.pickle 
   ```

   
