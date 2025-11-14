# CS 323 - Project 6 : Secure Decision Tree Evaluation

## Project Overview

This program securely evaluates a decision tree with continuous features 
with Secure Multiparty Computation. 
Part 1) Implement Secure Decision Tree Evaluation
Part 2) Test Accuracy with Non-secure and Secure Evaluation of Trained tree 
Part 3) Evaluate Runtime of Trees with Varying Depths 

- Long Pham and Tanvi Shegaonkar

## Project Structure
```

│── dt.py                                    # program file for part 1, part 2, and part 3
│── trained_diabetes_dt.txt                  # text file containing parameters for decision tree trained using scikit

    
```

## Instructions
# Ensure that trained_diabetes_dt.txt is in the same directory as dt.py
1. Specify your inputs in a file in the same format as trained_diabetes_dt.txt. Each line of the file should contain:
Line 1: Your tree's depth
Line 2: Your class labels, space-separated
Line 3: Your input formatting i.e. dataset schema (eg. if you typed (Pregnancies Glucose BloodPressure) then the first column 
        of the input_df should be Pregnancies)
Line 4: Your tree's attributes for internal nodes going from the root to its left child, then to its right child, then to the left child's left 
        child and so on and so forth. This tree should be a full decision tree so the number of nodes should be 2^d -1
Line 5: Your tree's class labels for leaf nodes going from the leftmost leaf to the rightmost leaf. The number of leaves should be 2^d
Line 6: Your thresholds for each attribute, in the same order as line 4. The tree will evaluate attribute <= threshold. 

2. Run python dt.py
   Part 1 classifies instances in diabetes.csv dataset using your tree
   Part 2 evaluates your tree's accuracy
   Part 3 tests runtime with respect to different tree depths, a visual plot is provided in Runtime.png