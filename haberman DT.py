# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 01:14:08 2020

@author: hp
"""
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

df=pd.read_csv('haberman.csv')

#Renaming the column names for ease
df.columns=['age_of_op','year_of_op','axillary_nodes','Survival']

#Replacing class label 2 with 0 as 0 better denotes Fatal surgery
a={2:0}
df.replace(a, inplace=True)

X=df.iloc[:,0:3].values
Y=df.iloc[:,3].values

#Splitting the data
X_train, X_test, Y_train, Y_test=tts(X, Y, test_size=0.2, random_state=42)

#Class weights assigned manually for balancing the data
w={1:26, 0:74}
classifier=tree.DecisionTreeClassifier(class_weight=w)
classifier.fit(X_train, Y_train)
y_pred=classifier.predict(X_test)

#Recall Score used to measure efficacy as it is more important to minimise false negatives
recall_score(Y_test,y_pred)
# 87% Recall Score

#Using XGBoost algorithm to improve upon the existing baseline score

param_grid={"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}
grid = GridSearchCV(xgb.XGBClassifier(), param_grid, refit = True, scoring='recall') 
grid.fit(X_train, Y_train)
print(grid.best_estimator_)
print(grid.score(X_test, Y_test))

#93% accuracy achieved












