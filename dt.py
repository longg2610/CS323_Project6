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


1) evaluation(x,w,classes,G,H,d)
    - can be called independently from Secure_Tree_Evaluation
    - implements secure evaluation with the following helper functions
        - parse(filename)
        - get_bit(number,i)
        - get_random_shares(value,)
        
2) Secure_Tree_Evaluation (parameter_file, input_df)
    - parses parameter_file specifying tree in format specified in README.md
    - evaluates decision tree specified for multiple instances stored in input_df


"""




"""
parse(filename):
Input: file containing tree parameters in specified format
Outputs:
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
            H[i] = features.index(H[i])        # construct mapping from internal node index to features
        G = [""] + (f.readline().split())
        for i in range (1, len(G)):
            G[i] = classes.index(G[i]) - 1     # construct mapping from leaf index to class labels
        w = [-1] + list(map(float, f.readline().split()))
    f.close()
    return [d, classes, features, H, G, w]

"""
get_bit(number, i)
- returns the i'th least significant bit of number, i starts from 1 
- input: number to get its bit from, index i of the bit (lsb is at i=1)
- output: bit i of number
"""
def get_bit(number, i):
    return ((number >> (i-1)) & 0x1)

"""
get_random_shares (value)
- randomly generates bitwise shares by XOR-ing the random value with value
input: int value
output: list of two bitwise shares of value 

"""
def get_random_shares(value):    
    A_share = random.sample(range(2), 1)[0]
    B_share = A_share ^ value
    return [A_share, B_share]


""" 
evaluation(x,w,classes,G,H,d)
- classify instance x given
- input:
    x: instance to be classified
    w: mapping of thresholds to internal nodes
    classes: list of class labels
    G: mapping of leaves index to class labels
    H: mapping of internal nodes index to attributes
    d: depth of tree
- output: x's classification
"""

def evaluation(x,w,classes,G,H,d):
    # get Alice's input H(i) corresponding to leaf i using non-private Oblivious Input Selection
    x_H = [-1]
    for attribute_index in H[1:]:
        x_H.append(x[attribute_index])

    # get comparison result for leaf i using non-private Distributed Comparison
    z_i = [-1]
    for i in range(1, len(H)):
        z_i.append(int(x_H[i] <= w[i]))
    z_i_shares = [get_random_shares(z) for z in z_i]
    
    alpha = math.ceil(math.log2(len(classes)-1))    # number of bits needed to represent all classes
    
    # contains r lists corresponding to r bits of the class label
    # each of which contains j pairs corresponding to the number of leaves
    # each pair is of form [Alice's share, Bob's share]
    sigma =  [[[None for _ in range(2)] for _ in range(2**d)] for _ in range(alpha)]    
    for j in range(2**d):
        b = G[j + 1]
        for r in range(1, alpha+1):
            y_j_r = [0, get_bit(b, r)]
            sigma[r-1][j] = y_j_r

        u = 1
        s = d
        while(s > 0):
            z_sum = z_i_shares[u].copy()        # z_u
            z_sum[1]  = (z_sum[1] + get_bit(j,s)) % 2   # add j_s to Bobby's bit -> z_u + j_s
            for r in range(1, alpha + 1):
                y_j_r = sigma[r-1][j]
                
                # multiply XOR of shares of y_j_r with XOR of shares of z_u + j_s, then break again into Alice's and Bob's shares
                sigma[r-1][j] = get_random_shares((y_j_r[0] ^ y_j_r[1]) * (z_sum[0] ^ z_sum[1]))

            # update u and s
            u = 2*u + get_bit(j, s)
            s -= 1

    label_bits = []     # contains bit representation of class label
    for r in range(1, alpha + 1):
        sigma_r = sigma[r-1].copy()
        bit = sum([x[0]^x[1] for x in sigma_r])
        label_bits.append(bit)
    
    # convert bit representation to class index
    class_index = 0
    for i in range(len(label_bits)):
        class_index += (2**i) * label_bits[i]

    print("Instance", x[1:], "is classified as", classes[class_index + 1])
    return class_index



def Secure_Tree_Evaluation (parameter_file, input_df):
    tree_parameters = parse(parameter_file)
    d, classes, features, H, G, w = tree_parameters
    print("Input order:\n", features[1:])
    output = []
    
    #iterates over input_df and evaluates tree for each instance
    for i in range(len(input_df)):
        instance = input_df.iloc[i].tolist()
        instance.insert(0,-1)   # dummy value at the front
        class_label = evaluation(instance,w,classes,G,H,d)
        output.append(class_label)
    result = pd.Series(output)
    return(result)
 
 
""" 
Part 2: Evaluating Accuracy

We use scikit-learn to create a trained decision tree
of depth 3 for the diabetes.csv dataset.
The parameters for the trained tree are encoded in trained_diabeters_dt.txt. 

We evaluate the accuracy of the decision tree in a 
non-secure setting and secure setting.

Identical results for accuracy are produced in both settings, 
verifying that secure evaluation was implemented correctly in Part 1. 
"""
 
file = pd.read_csv("diabetes.csv")
df = pd.DataFrame(file)
if "SkinThickness" in df.columns:          
    df = df.drop(columns=["SkinThickness"])
    

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']
target_names = ['Outcome']
class_labels = ["No", "Yes"]

# Creates dataset of feature values and of target values
X = df[feature_names]
y = df[target_names]


# partition dataset into training data and testing data
# 40% of data reserved for testing
# Seeded for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=21)

# Creates decision tree of depth 3, seeded for reproducibility
dtree = DecisionTreeClassifier(criterion="entropy", random_state= 21, max_depth=3)

# Trains dtree
dtree.fit(X_train, y_train)

#Evaluates dtree in non-secure setting
y_pred = dtree.predict(X_test)

#Evaluates dtree in Secure Setting
#trained_diabetes_dt.txt contains parameters for dtree
y_pred_secure = Secure_Tree_Evaluation ("trained_diabetes_dt.txt", X_test)

print()
print("Part 2: Evaluating Accuracy")

print("Non-secure Tree Evaluation Accuracy: ", round(accuracy_score(y_test, y_pred)*100,2))
print("Secure Tree Evaluation Accuracy: ", round(accuracy_score(y_test, y_pred_secure)*100,2))


 
""" 
Part 3: Evaluating Runtime

We evaluate how the depth of a decision tree impacts runtime 
of decision tree evaluation in a secure setting. 

We randomly generate nodes, weights, and class labels to 
create full decision trees of varying depths. 
Runtime is measured in milliseconds for evaluation
of each tree for the same instance


"""
print()
print("Part 3: Evaluating Runtime")
runtime = []
classes = ["", "Yes", "No"]

# instance to be tested at each depth
x = [-1, 4, 119, 69, 19, 30, 0.1, 29]


depths = [3,5,7,9,11,15]
depths = [3,4,5,6,7,8,9,10,11,12,13,14,15]
for d in depths:
    num_leaf = 2**d
    num_internal = num_leaf -1

    # Creates full dtree of depth d
    # with random nodes, weights, and leaf class labels
    H = random.choices(range(8), k=num_internal)
    w = random.choices(range(100),k = num_internal)
    G = random.choices(range(2), k = num_leaf)
    
    # Ensures indexing begins at 1 
    H.insert(0,-1)
    G.insert(0,-1)
    w.insert(0,-1)
    
    #Executes secure evaluation and collects runtime data
    t_0 = time()
    evaluation(x,w,classes,G,H,d)
    final_time = round(((time()-t_0)*1000),2)
    runtime.append(final_time)
    print("Runtime d =", d, ": ", final_time, " ms")

def plot(depths, runtime):
    # Create line chart
    plt.figure(figsize=(10,7))
    plt.plot(depths, runtime, marker='o', linestyle='-', color='blue', label='Depths vs Runtime')

    # Labels and title
    plt.xlabel("Depth")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtimes with different Decision Tree Depths")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig("./Runtime.png")
plot(depths, runtime)
