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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=21)

dtree = DecisionTreeClassifier(criterion="entropy", random_state= 21, max_depth=3)

dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)


plt.figure(figsize=(15, 10))
tree.plot_tree( dtree,  filled=True, rounded=True, feature_names = feature_names, class_names= class_labels)
plt.show()



