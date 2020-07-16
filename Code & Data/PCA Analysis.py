# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:37:33 2020

@author: Mickey

A scipt that is used to analyse and visualise what the optimal number of 
n_components in a principal component analysis would be with the dataset
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Sets the style for the plots
sns.set_style("darkgrid")

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

# Fits PCA to the data
pca = PCA()
pca.fit(X)


# Prints info about the explained variance of the data, the explained variance 
# ratio and its cumulative sum 
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())
pca_variance = pca.explained_variance_ratio_.cumsum()


# Creates a plot that shows the cumulative sum of the explained variance 
# against different numbers of principle components
fig, ax = plt.subplots(figsize=(15,10))
#plt.title("PCA")
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components', fontsize=15)
plt.ylabel('Cumulative explained variance', fontsize=15);
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
