
## Decision Trees

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## Load data
df = pd.read_csv("prepped_churn_data.csv")
df.head(5)

## Create features and targets

features = df.drop('Churn', axis=1)
targets = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(features, targets, stratify = targets, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

dt.get_depth()

f = plt.figure(figsize = (15,15))
_ = plot_tree(dt,fontsize=8,feature_names = features.columns, filled=True)

dt=DecisionTreeClassifier(max_depth=2)
dt.fit(x_train,y_train)

print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))

f = plt.figure(figsize=(8,8))
_ = plot_tree(dt,fontsize=10,feature_names=features.columns,filled=True)

## Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=5, random_state=42)
rfc.fit(x_train,y_train)

print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))

#If overfitting occurs again, reduce the max_depth and repeat the above code

## Tune the max_features

import math
math.sqrt(x_train.shape[1])

rfc = RandomForestClassifier(max_depth=2, max_features=7, random_state=42)
rfc.fit(x_train,y_train)
print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))

# To automatically tune these hyperparameters like max_features and max_depth we can use hyperparameter search like GRidSearchCV or Bayesian search

## Feature Selection
import seaborn as sns

f = plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)

## Feature importances (Can also use dt)

from scikitplot.estimators import plot_feature_importances

plot_feature_importances(rfc,feature_names = features.columns, x_tick_rotation=90)

new_features = features.drop([], axis =1) # Drop the bottom features

x_train,x_test,y_train,y_test=train_test_split(new_features, targets, stratify=targets, random_state=42)

rfc = RandomForestClassifier(max_depth=2, max_features=7, random_state=42)
rfc.fit(x_train, y_train)
print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))

plot_feature_importances(rfc, feature_names=new_features.columns, x_tick_rotation=90)
