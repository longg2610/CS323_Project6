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
│── input.txt                                # Shamir's Method Implementation. Should be in same directory
│── trained_diabetes_dt.txt                  # text file containing parameters for decision tree trained using scikit

Everything below are generated files happening during the execution of the code:
│── shamir_runtime_avg.csv        # This is the file contains the average runtime across three trials of Shamir's Method
│── paillier_run_time_avg.csv     # This is the file contains the average runtime across three trials of Paillier's Method
    
```

## Instructions
# Ensure that the Paillier_Final.ipynb, Shamir_Final.ipynb, and Runtime_Comparison.ipynb are in the same directory 
# Please run Shamir_Final.ipynb and Paillier_Final.ipynb before running Runtime_Comparison.ipynb