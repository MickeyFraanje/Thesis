# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:22:39 2020

@author: Mickey

The clustering model used in experiment 2
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import homogeneity_score
import matplotlib.pyplot as plt

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

# Files that include region, income group and HDI
meta_file = "Metadata.csv"
hdi_file = "Human Development Index (HDI).csv"

# Creates a dataframe for the meta data
meta = pd.read_csv(meta_file)[["TableName", "Country Code", "Region", "IncomeGroup"]]

# Transforms the HDI file into a usable dataframe which can be used to 
# calculate the mean HDI per country
hdi_full = pd.read_csv(hdi_file)
hdi_full = hdi_full[["Country", "2016", "2017", "2018"]].dropna()
hdi_full["2016"] = hdi_full["2016"][hdi_full["2016"] != '..']
hdi_full["2017"] = hdi_full["2017"][hdi_full["2017"] != '..']
hdi_full["2018"] = hdi_full["2018"][hdi_full["2018"] != '..']
hdi_full["2016"] = pd.to_numeric(hdi_full["2016"])
hdi_full["2017"] = pd.to_numeric(hdi_full["2017"])
hdi_full["2018"] = pd.to_numeric(hdi_full["2018"])

# Creates a dataframe that includes the mean HDI per country
hdi = pd.DataFrame()
hdi["Average HDI"] = hdi_full.drop('Country', axis=1).apply(lambda x: x.mean()
                                                            , axis=1)
hdi["Country"]= hdi_full["Country"]

# Creates categories of the continuous HDI data so that a homogeneity score 
# can be calculated after clustering the data
hdi["Average HDI"][hdi["Average HDI"].between(0.3, 0.399)] = 3
hdi["Average HDI"][hdi["Average HDI"].between(0.4, 0.499)] = 4
hdi["Average HDI"][hdi["Average HDI"].between(0.5, 0.599)] = 5
hdi["Average HDI"][hdi["Average HDI"].between(0.6, 0.699)] = 6
hdi["Average HDI"][hdi["Average HDI"].between(0.7, 0.799)] = 7
hdi["Average HDI"][hdi["Average HDI"].between(0.8, 0.899)] = 8
hdi["Average HDI"][hdi["Average HDI"].between(0.9, 0.999)] = 9

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

# Merges the data and the meta dataframes
merged_data = pd.merge(data, meta,left_on=['Country'], right_on = ['TableName'], how = 'left').dropna()


"""
Dendrogram code based on the example provided on Scikit-learn's website.

Title: <Plot Hierarchical Clustering Dendrogram>
Author: <Mathew Kallada, Andreas Mueller>
Date: <2019>
Code version: <Python 3.6>
Availability: <https://scikit-learn.org/stable/auto_examples/cluster/
               plot_agglomerative_dendrogram.html>
"""
def plot_dendrogram(model,**kwargs):
    # Creates linkage matrix and then plots the dendrogram

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the dendrogram
    dendrogram(linkage_matrix,  **kwargs)

# Creates a dataframe that only includes the variables necessary for clustering
X = data[data.columns.difference(["Country", "CountryCode", "Country Name", 
                                  "Country Code"])]

# Takes the mean per country so that each datapoint is a single country
data = data.groupby("Country", as_index = False).mean()

# Splits the data into a data matrix (X) and a label vector (y) for use by the
# clustering model
X = data[data.columns.difference(["Unnamed: 0", "Country", 
                                  "CountryCode"])].set_index(data["Country"])
y = X.index

# Merges the data and the meta dataframes
merged_data = pd.merge(data, meta,left_on=['Country'], 
                       right_on = ['TableName'], how = 'left').fillna('NaN')
# Merges the merged data and the HDI dataframes
merged_data = pd.merge(merged_data, hdi, left_on=['Country'], 
                       right_on = ['Country'], how = 'left').fillna(0)

# Creates a variable for each important column
region = merged_data["Region"]
income_group = merged_data["IncomeGroup"]
hdi_score = merged_data["Average HDI"]


# Fits a scaler and scales the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X = scaler.transform(X)


# Fits a model to create a dendrogram with
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)

# Creates and plots the dendrogram
fig, ax = plt.subplots(figsize=(20,10))
#plt.title('Hierarchical Clustering Dendrogram', fontsize = 20)
plot_dendrogram(model, labels = y, ax = ax, p=30, color_threshold=4)
plt.ylabel("Distance", fontsize=15)
plt.xlabel("Countries", fontsize=15)
plt.axhline(y=4, c='grey', lw=2, linestyle='dashed')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Fits another model with a set amount of clusters to test the cluster 
# homogeneity scores
model = AgglomerativeClustering(n_clusters = 17)
cluster_data = pd.DataFrame()

# Baseline that consists of an array of 7 integers to simulate the other 
# columns that are used to test similarity
baseline = np.random.randint(0,6,65)

# Calculates and prints the homogeneity scores
print(homogeneity_score(y, model.fit_predict(X)))
print("Baseline homogeneity: ", homogeneity_score(baseline, 
                                                  model.fit_predict(X)))
print("Region homogeneity: ", homogeneity_score(region, 
                                                model.fit_predict(X)))
print("Income group homogeneity: ", homogeneity_score(income_group, 
                                                      model.fit_predict(X)))
print("HDI Score homogeneity: ", homogeneity_score(hdi_score, 
                                                   model.fit_predict(X)))
# Creates a data frame containing all relevant variables
cluster_data["Cluster"] = model.fit_predict(X)
cluster_data["Country"] = data["Country"]
cluster_data["Region"] = region
cluster_data["IncomeGroup"] = income_group
cluster_data["HDI Score"] = hdi_score

# Creates a dataframe containing all cluster data
cluster_data = cluster_data.sort_values("Cluster", ascending = True)