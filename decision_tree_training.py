import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
from time import time


file = pd.read_csv("diabetes.csv")
df = pd.DataFrame(file)
if "SkinThickness" in df.columns:          
    df = df.drop(columns=["SkinThickness"])



feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']
target_names = ['Outcome']
class_labels = ["No", "Yes"]
X = df[feature_names]
y = df[target_names]

print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

dtree = DecisionTreeClassifier(criterion="entropy", random_state= 21, max_depth=3)

dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)


plt.figure(figsize=(15, 10))
tree.plot_tree( dtree,  filled=True, rounded=True, feature_names = feature_names, class_names= class_labels)
plt.show()



"""
d = 3
# 
# all inputs are indexed from 1. Thus, a placeholder value is inserted at index 0

# class labels
classes = ["", "Yes", "No"]
    
# H maps internal nodes to feature indices, H(1) = index of Pregnancies in features array
H = [-1, 6, 2, 4, 3, 1, 5, 7]  
# w contains the thresholds/ weights for each internal node in H
w = [-1 , 0.15, 120, 20, 70, 5, 28, 30]
# G maps leaf indices to class label indices
G = [-1,1,0,1,0,1,0,1,0] 

# features = ["", "Pregnancies" , "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction" , "Age"]     # input follows this format
# x = (-1, 4, 119, 69, 19, 30, 0.1, 29)   


# Class Example instance
# d = 2
# G = [-1,0,1,0,1]
# H = [-1,2,1,3]
# w = [-1,1,1,1]
# x = (-1,0,0,0)

"""



