# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:51:04 2020

@author: Mickey

This script creates a boxplot and dataframe that includes the mean and 
standard deviation of the country references per referenced country
"""

import pandas as pd
import numpy as np
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

# Calculates the mean of each column
mean = data.mean(axis= 0)
mean = pd.to_numeric(mean)

# Calculates the standard deviation for each column
std = data.std(axis= 0)
std = pd.to_numeric(std)

# Creates a dataframe containing both the mean and the standard deviation of
# each column in a way that can be sorted by mean
stat = pd.concat([mean, std], axis=1)
stat = stat.sort_values(by=[0], ascending = False)

# Sets the style of the plot
sns.set_style('whitegrid')

# Build the boxplots
fig, ax = plt.subplots(figsize = (15, 10))
ax.grid(False)
data.boxplot(column = ["SELF", "US", "UK", "RS", "FR", "GM", "CH", "IT", "SP", "TU", "JA"], 
             showmeans = False, flierprops = dict(marker='_', markerfacecolor='none', markersize=2,
             linestyle='none'), widths = 0.5, boxprops = dict(linestyle='-', linewidth=2, color = 'black'),
             medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick'), 
             whiskerprops = dict(linestyle='--', linewidth=2, color = 'black'), 
             capprops = dict(linewidth=2.5)
)
plt.ylabel("Mean references", fontsize=15)
plt.xlabel("Countries and self-reference columns", fontsize=15) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim((0,1))

# Displays the plot
plt.show()

