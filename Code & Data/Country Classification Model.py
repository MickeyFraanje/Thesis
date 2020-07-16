# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 00:29:54 2020

@author: Mickey
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


# Each of the files used the dataset
file = "2016 COUNTRIES.csv"
file2 = "2017 COUNTRIES.csv"
file3 = "2018 COUNTRIES.csv"
file4 = "2019 COUNTRIES.csv"

# Reads the data from their csv files and appends them into a single dataframe
data = pd.read_csv(file)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)
data4 = pd.read_csv(file4)
data = data.append(data2).fillna(0)
data = data.append(data3).fillna(0)
data = data.append(data4).fillna(0)

def order_data(data, country_threshold):
    # This function orders and cleans the dataset in order to increase 
    # readability and model effectiveness.
    #
    # Firstly it rearranging the columns so that "Country" is in front for 
    # reader clarity.
    cols = list(data)
    cols.insert(0, cols.pop(cols.index('Country')))
    data = data[cols]

    # Counts the amount of instances of each country and delete nations with a 
    # frequency that is below the threshold
    print(data["Country"].value_counts())

    # Remove items less than or equal to threshold
    threshold = country_threshold  
    vc = data["Country"].value_counts()
    vals_to_remove = vc[vc <= threshold].index.values
    data["Country"].loc[data["Country"].isin(vals_to_remove)] = None
    data = data[data.Country != None]


    data = data.dropna()

    print(data["Country"].value_counts())
    return data

data = order_data(data, 199)

# Splits the data into a data matrix (X) and a label vector (y) for use by the
# supervised models
X = data[data.columns.difference(["Country", "CountryCode"])]
y = data["Country"]


# Makes a split of the data with no preprocessing done for use in the 
# feature importance analysis
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Random Forest is used to perform a feature importance analysis
print("Random Forest Feature Importance Analysis:")

rfc = RandomForestClassifier()
rfc = rfc.fit(X, y)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
print("Testing score: ",accuracy_score(  y_test, y_pred))

# Sets the style for the plots
sns.set_style('whitegrid')
# Visualisation of the feature importance analysis
fig, ax = plt.subplots(figsize=(20,40))
feat_importances = pd.Series(rfc.feature_importances_, index=X.columns)
feat_importances.nlargest(100).plot(kind='barh', ax = ax, 
                                   color = "cornflowerblue")
plt.ylabel("Features", fontsize=15)
plt.xlabel("Importance", fontsize=15)            
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.title("Feature importance analysis on country reference data", 
#          fontsize = 20)
plt.show()

feature_importance = rfc.feature_importances_


# PCA used for dimensionality reduction
pca = PCA(n_components = 50)
pca.fit(X)
X_pro = pca.transform(X)

# Sets a scaler and scales the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_pro)
X_pro = scaler.transform(X_pro)

# Splits the data into a train and a test set
X_train, X_test, y_train, y_test = train_test_split(X_pro, y)


# Trains a dummy classifier to act as baseline
print("Baseline dummy classifier:")

baseline = DummyClassifier(strategy= "uniform").fit(X_train, y_train)
y_pred = baseline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Testing score: ",accuracy_score(  y_test, y_pred))
y_pred = baseline.predict(X_train)
print("Training score: ", accuracy_score(  y_train, y_pred))


# Trains a KNN Classifier
print("K-Nearest Neighbours Classifier:")

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print("Testing score: ",accuracy_score(  y_test, y_pred))
y_pred = knn.predict(X_train)
print("Training score: ", accuracy_score(  y_train, y_pred))


# Trains a Naive Bayes Classifier
print("Naive Bayes Classifier:")

gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred))
print("Testing score: ", accuracy_score(  y_test, y_pred))
y_pred = gnb.predict(X_train)
print("Training score: ", accuracy_score(  y_train, y_pred))


# Trains a Support Vector Machines Classifier
print("Support Vector Machines Classifier:")

svm = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='auto', 
          kernel='linear', max_iter=-1, probability=False, random_state=None, 
          shrinking=True, tol=0.001, verbose=False).fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))
print("Testing score: ", accuracy_score(  y_test, y_pred))
y_pred = svm.predict(X_train)
print("Training score: ", accuracy_score(  y_train, y_pred))

# Trains a Random Forest Classifier
print("Random Forest Classifier:")

rfc = RandomForestClassifier(max_depth = 8)
rfc = rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
print("Testing score: ",accuracy_score(  y_test, y_pred))
y_pred = rfc.predict(X_train)
print("Training score: ",accuracy_score(  y_train, y_pred))


