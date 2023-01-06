# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:48:35 2023

@author: Rahul
"""
import numpy as np
import pandas as pd
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Decision Trees\\Company_Data.csv")
df

df.describe()
df.info()
df.dtypes

# Converting Target variable 'Sales' into categories Low, Medium and High.
df['Sales'] = pd.cut(x=df['Sales'],bins=[0, 6, 12, 17], labels=['Low','Medium', 'High'], right = False)
df['Sales']

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))

# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

#pairwise plot of all the features
sns.pairplot(df)
plt.show()

# checking count of categories for categorical columns colums
sns.countplot(df['ShelveLoc'])
plt.show()

sns.countplot(df['Urban'])
plt.show()

sns.countplot(df['US'])
plt.show()

sns.countplot(df['Sales'])
plt.show()

#Label Encoder
from sklearn.preprocessing import LabelEncoder
LE =LabelEncoder()
df.iloc[:,6] = LE.fit_transform(df.iloc[:,6])
df.iloc[:,9] = LE.fit_transform(df.iloc[:,9])
df.iloc[:,10] = LE.fit_transform(df.iloc[:,10])
df.iloc[:,0] = LE.fit_transform(df.iloc[:,0])
df.head()

#splitting the data into x and y
x = df.drop('Sales',axis=1)
y = df["Sales"]

#Data Partition
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=(40))

model = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
model.fit(x_train,y_train) 

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("training score:",ac1.round(2))
ac2 = accuracy_score(y_test,y_pred_test)
print("test score:",ac2.round(2))

plt.figure(figsize=(18,10)) 
tree.plot_tree(model,filled=True)
plt.title('Decision tree using Entropy',fontsize=22)
plt.show() 

model1 = DecisionTreeClassifier(criterion = 'gini',max_depth=4)
model1.fit(x_train,y_train) 

y_pred_train = model1.predict(x_train)
y_pred_test = model1.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("training score:",ac1.round(2))
ac2 = accuracy_score(y_test,y_pred_test)
print("test score:",ac2.round(2))

plt.figure(figsize=(18,10)) 
tree.plot_tree(model,filled=True)
plt.title('Decision tree using gini',fontsize=22)
plt.show() 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('Random Forest Classifier', RandomForestClassifier(max_depth=0.7)))

for title, modelname in models:
    modelname.fit(x_train, y_train)
    
y_pred = modelname.predict(x_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print(title,"Accuracy: %.2f%%" % (accuracy * 100.0))









































