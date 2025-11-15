# CS 323 - Project 6 : Secure Decision Tree Evaluation

## Project Overview

This program securely evaluates a decision tree with continuous features 
with Secure Multiparty Computation. 
Part 1) Implement Secure Decision Tree Evaluation
Part 2) Test Accuracy with Non-secure and Secure Evaluation of Trained tree 
Part 3) Evaluate Runtime of Trees with Varying Depths 
Part 4) Testing Area for Dr. Truex

- Long Pham and Tanvi Shegaonkar

## Project Structure
```

│── dt.py                                    # program file for part 1, part 2, part 3, and part 4
│── trained_diabetes_dt.txt                  # text file containing parameters for trained decision tree in Part 2
│── sample_tree.txt                          # text file containing parameters for sample tree in Part 4
│── diabetes.csv                             # csv file containing data for diabetes dataset
                                             source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download


Outputs
│── Runtime.png                              # png file containing graph generated in Part 3


    
```

## Instructions
# Ensure that diabetes.csv, trained_diabetes_dt.txt, and sample_tree.txt is in the same directory as dt.py

2. Run python dt.py
   Part 1 classifies instances in diabetes.csv dataset using your tree
   Part 2 evaluates your tree's accuracy
   Part 3 tests runtime with respect to different tree depths, a visual plot is provided in Runtime.png
   Part 4 provides a testing areas to evaluate different trees or instances

** to remove print statements such as "Instance [0, 0, 0] is classified as c1", comment out line 162 **

To evaluate a new decision tree for an instance or dataset, Part 4 can be used. 
More specific instructions can be found in comments for Part 4. 

1. If evaluating one instance for a certain tree, specify tree parameters and instance inline as described in Part 4.
    This will be evaluated using the evalution() method

2. If evaluating a dataset for a tree, specify tree parameters in the same format as sample_tree.txt and trained_diabetes_dt.txt
    The dataset should have columns in the same order as Line 3 of the tree parameter file.

    Each line of the file should contain:

Line 1: Your tree's depth
Line 2: Your class labels, space-separated
Line 3: Your input formatting i.e. dataset schema (eg. if you typed (Pregnancies Glucose BloodPressure) then the first column 
        of the input_df should be Pregnancies)
Line 4: Your tree's attributes for internal nodes going from the root to its left child, then to its right child, then to the left child's left 
        child and so on and so forth. This tree should be a full decision tree so the number of nodes should be 2^d -1
Line 5: Your tree's class labels for leaf nodes going from the leftmost leaf to the rightmost leaf. The number of leaves should be 2^d
Line 6: Your thresholds for each attribute, in the same order as line 4. The tree will evaluate attribute <= threshold. 

This tree will be evaluated using the Secure_Tree_Evaluation method()

Analysis was uploaded to Canvas seperately in Project_6.pdf 