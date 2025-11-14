'''
Long Pham and Tanvi Shegaonkar
Project 6 : Secure Decision Tree Evaluation
11/14/2025

This program securely evaluates a decision tree with continuous features 
with Secure Multiparty Computation. 
Part 1) Implement Secure Decision Tree Evaluation
Part 2) Test Accuracy with Non-secure and Secure Evaluation of Trained tree 
Part 3) Evaluate Runtime of Trees with Varying Depths

'''
import random
import math
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from time import time

""" 
Part 1: Implement Secure Decision Tree Evaluation
Input - file specifying parameters

"""
    
"""
# d = 3
# # 
# # all inputs are indexed from 1. Thus, a placeholder value is inserted at index 0

# # class labels
# classes = ["", "Yes", "No"]
    
# # H maps internal nodes to feature indices, H(1) = index of Pregnancies in features array
# H = [-1, 6, 2, 4, 3, 1, 5, 7]  
# w contains the thresholds/ weights for each internal node in H
# w = [-1 , 0.15, 120, 20, 70, 5, 28, 30]
# G maps leaf indices to class label indices
# G = [-1,1,0,1,0,1,0,1,0] 

# features = ["", "Pregnancies" , "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction" , "Age"]     # input follows this format
# x = (-1, 4, 119, 69, 19, 30, 0.1, 29)   


# Class Example instance
# d = 2
# G = [-1,0,1,0,1]
# H = [-1,2,1,3]
# w = [-1,1,1,1]
# x = (-1,0,0,0)

"""

"""
parse(filename):
Input: file containing tree parameters in specified format
Ouputs:
    - int d = depth of tree
    - string list classes = list of class labels
    - string list features = list of features
    - list H maps internal nodes to feature indices, e.g. H(1) = index of Pregnancies in features array
    - G maps leaf indices to class label indices
    - list w contains the thresholds/ weights for each internal node in H
"""
def parse(filename):
    with open(filename, 'r') as f:
        d = int(f.readline())
        classes = [""] + (f.readline().split())
        features = [""] + (f.readline().split())
        H = [-1] + f.readline().split()
        for i in range (1, len(H)):
            H[i] = features.index(H[i])     # construct mapping
        G = [""] + (f.readline().split())
        for i in range (1, len(G)):
            G[i] = classes.index(G[i]) - 1     # construct mapping
        w = [-1] + list(map(float, f.readline().split()))
    f.close()
    return [d, classes, features, H, G, w]

"""
get_bit(number, i)
- returns the i'th least significant bit of number, i starts from 1 
"""
def get_bit(number, i):
    return ((number >> (i-1)) & 0x1)

"""
get_random_shares (value, l)
- randomly generates bitwise shares by XOR-ing the random value with value
input: int value
       int l = bit length

output: list of two bitwise shares of value 

"""
def get_random_shares(value):    
    A_share = random.sample(range(2), 1)[0]
    B_share = A_share ^ value
    return [A_share, B_share]

""" 
get_z_H_values()
for each non-leaf node, this function evaluates if the value
of input vector is less than or equal to the threshold of 
its associated node. 
Outputs: bitwise shares of length l
"""
def get_z_H_values(x,w,H): 
    x_H = [-1]
    for attribute_index in H[1:]:
        x_H.append(x[attribute_index])

    z_i = [-1]
    for i in range(1, len(H)):
        z_i.append(int(x_H[i] <= w[i]))
    # print(z_i)
    z_i_shares = [get_random_shares(z) for z in z_i]
    return z_i_shares

def evaluation(x,w,classes,G,H,d):
    z_i_shares = get_z_H_values(x,w,H)           # z_i is good
    alpha = math.ceil(math.log2(len(classes)-1))
    sigma =  [[[None for _ in range(2)] for _ in range(2**d)] for _ in range(alpha)]
    for j in range(2**d):
        b = G[j + 1]
        for r in range(1, alpha+1):
            y_j_r = [0, get_bit(b, r)]
            sigma[r-1][j] = y_j_r

        u = 1
        s = d
        while(s > 0):
            z_sum = z_i_shares[u].copy()
            z_sum[1]  = (z_sum[1] + get_bit(j,s)) % 2   # add j_s to Bobby's bit -> z_u + j_s
            for r in range(1, alpha + 1):
                y_j_r = sigma[r-1][j]
                
                # multiply XOR of shares of y_j_r with XOR of shares of z_u + j_s  
                sigma[r-1][j] = get_random_shares((y_j_r[0] ^ y_j_r[1]) * (z_sum[0] ^ z_sum[1]))

            # update u and s
            u = 2*u + get_bit(j, s)
            s -= 1

    # print(sigma)

    label_bits = []
    for r in range(1, alpha + 1):
        sigma_r = sigma[r-1].copy()
        class_index_bit = sum([x[0]^x[1] for x in sigma_r])
        label_bits.append(class_index_bit)
        # print("bit ", r, ": ", class_index_bit)
    
    class_index = 0
    for i in range(len(label_bits)):
        class_index += (2**i) * label_bits[i]

    print("Instance", x, "is classified as", classes[class_index + 1])
    return(class_index)



def Secure_Tree_Evaluation (parameter_file, input_df):
    
    tree_parameters = parse(parameter_file)
    print(tree_parameters)
    d, classes, features, H, G, w = tree_parameters
   
    output = []
    for i in range(len(input_df)):
        instance = input_df.iloc[i].tolist()
        instance.insert(0,-1)
        print("X ", instance)

        class_label = evaluation(instance,w,classes,G,H,d)
        output.append(class_label)
    result = pd.Series(output)
    return(result)
 
""" 
Part 2: 

"""
 
file = pd.read_csv("diabetes.csv")
df = pd.DataFrame(file)
if "SkinThickness" in df.columns:          
    df = df.drop(columns=["SkinThickness"])
print(df.shape)
    
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']
target_names = ['Outcome']
class_labels = ["No", "Yes"]
X = df[feature_names]
y = df[target_names]

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=21)

dtree = DecisionTreeClassifier(criterion="entropy", random_state= 21, max_depth=3)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
y_pred_secure = Secure_Tree_Evaluation ("trained_diabetes_dt_input.txt", X_test)

print(accuracy_score(y_test, y_pred)*100)
print(accuracy_score(y_test, y_pred_secure)*100)



"""
Part 3: evaluating with multiple classes, instances are found in instances.txt
"""

d, classes, features, H, G, w = parse("input.txt")
with open('instances.txt', 'r') as f:
    for line in f:
        x = [-1] + list(map(float, (line.split())))
        evaluation(x, w, classes, G, H, d)
        