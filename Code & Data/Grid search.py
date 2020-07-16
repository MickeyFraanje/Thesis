# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:31:34 2020

@author: Mickey
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


# Each of the files used the dataset
# Change this to use optimise models on different datasets
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
    cols.insert(0, cols.pop(cols.index("Country")))
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


# Performs a gridsearch for KNN
print("K-Nearest Neighbours Classifier:")
knn = KNeighborsClassifier()
print("Fitting the classifier to the training set")
param_grid = {"n_neighbors":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15, 16, 17, 20,
                             ]}
# Fits the model
knn_model = GridSearchCV(knn, param_grid=param_grid)
knn_model.fit(X_train,y_train)
print("Best estimator found by grid search: ")
print(knn_model.best_estimator_)


# Performs a gridsearch for Random Forest
print("Random Forest Classifier: ")
rfc = RandomForestClassifier()
print("Fitting the classifier to the training set")
param_grid = {"max_depth":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15, 16, 17, 20,
                             ]}
# Fits the model
rfc_model = GridSearchCV(rfc, param_grid=param_grid)
rfc_model.fit(X_train,y_train)
print("Best estimator found by grid search:")
print(rfc_model.best_estimator_)


# Performs a gridsearch for SVC
print("Support Vector Machines Classifier:")

print("Fitting the classifier to the training set")
param_grid = {'C': [0.01, 0.1, 1, 10, 100], "kernel": ["rbf", "linear"]}
svc_model = GridSearchCV(SVC(class_weight='balanced'), param_grid)
# Fits the model
svc_model.fit(X_train, y_train)
print("Best estimator found by grid search:")
print(svc_model.best_estimator_)


# Final print at the end of the best estimators for readability
print("KNN: ", knn_model.best_estimator_)
print("Random Forest: ", rfc_model.best_estimator_)
print("SVC: ", svc_model.best_estimator_)