# -*- coding: utf-8 -*-
"""
@author: Mickey Fraanje

HDI model used in experiment 3
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Each of the files used the dataset
file = "2016 COUNTRIES.csv"
file2 = "2017 COUNTRIES.csv"
file3 = "2018 COUNTRIES.csv"

# Reads the data from their csv files and appends them into a single dataframe
data = pd.read_csv(file)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)

# Reads the HDI file
hdi_file = "Human Development Index (HDI).csv"
hdi_full = pd.read_csv(hdi_file)

# Creates a usable dataframe for the HDI data
hdi2016 = hdi_full[["Country", "2016"]].dropna()
hdi2017 = hdi_full[["Country", "2017"]].dropna()
hdi2018 = hdi_full[["Country", "2018"]].dropna()

hdi2016.columns = ["Country", "HDI"]
hdi2017.columns = ["Country", "HDI"]
hdi2018.columns = ["Country", "HDI"]

# Merges the data with the HDI dataset of the appropriate year
data = pd.merge(data, hdi2016, left_on=['Country'], right_on = ['Country'], 
                how = 'left').dropna()
data2 = pd.merge(data2, hdi2017, left_on=['Country'], right_on = ['Country'], 
                 how = 'left').dropna()
data3 = pd.merge(data3, hdi2018, left_on=['Country'], right_on = ['Country'], 
                 how = 'left').dropna()

# Combines the dataset into one
data = data.append(data2, sort = False).fillna(0)
data = data.append(data3, sort = False).fillna(0)


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

data = order_data(data, 149)

# Splits the data into a data matrix (X) and a label vector (y) for use by the
# supervised models
X = data[data.columns.difference(["Tone","Activity Reference Density",  
                                  "Country", "CountryCode", "TableName",
                                  "Country Code", "Region", "IncomeGroup",
                                  "HDI", "Country Name"])]
y = data["HDI"]


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


# Creates and trains the baseline model and prints its results
print("Baseline")

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
print("Train Score:")
print(baseline.score(X_train, y_train))
print("Test Score:")
print(baseline.score(X_test, y_test))
y_pred = baseline.predict(X_train)
print("MSE Train:")
print(mean_squared_error(y_train, y_pred))
print("MSE Test:")
y_pred = baseline.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print("R2 Score:")
print(r2_score(y_test, y_pred))
print(" ")

dummy_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Trains the ridge regression model and prints its results
print("Ridge Regression")

ridge = Ridge().fit(X_train, y_train)
print("Train Score:")
print(ridge.score(X_train, y_train))
print("Test Score:")
print(ridge.score(X_test, y_test))
y_pred = ridge.predict(X_train)
print("MSE Train:")
print(mean_squared_error(y_train, y_pred))
print("MSE Test:")
y_pred = ridge.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print("R2 Score:")
print(r2_score(y_test, y_pred))
print(" ")

ridge_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Trains the random forest model and prints its results
print("Random Forest Regressor")

forest = RandomForestRegressor(max_depth = 5).fit(X_train, y_train)
print("Train Score:")
y_pred = forest.predict(X_train)
print(forest.score(X_train, y_train))
print("MSE Train:")
print(mean_squared_error(y_train, y_pred))
print("Test Score:")
print(forest.score(X_test, y_test))
y_pred = forest.predict(X_test)
print("MSE Test:")
print(mean_squared_error(y_test, y_pred))
print("R2 Score:")
print(r2_score(y_test, y_pred))
print(" ")

forest_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Trains the SVR model and prints its results
print("SVR")

svr = SVR().fit(X_train, y_train)
print("Train Score:")
print(svr.score(X_train, y_train))
print("Test Score:")
print(svr.score(X_test, y_test))
y_pred = svr.predict(X_train)
print("MSE Train:")
print(mean_squared_error(y_train, y_pred))
y_pred = svr.predict(X_test)
print("MSE Test:")
print(mean_squared_error(y_test, y_pred))
print("R2 Score:")
print(r2_score(y_test, y_pred))
print(" ")

svr_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
