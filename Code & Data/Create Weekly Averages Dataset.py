# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:44:52 2020

@author: Mickey

Script that creates the country reference dataset used in the study
"""

#Records the time it takes to run the code
import time
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from calendar import monthrange

start_time = time.time()

# Lists in which dataframes are collected in order to be concatenated later
data_list = []
week_list = []
final_data_list = []

# Specify which year you want to extract here:
year = "2019"

# Goes through 4 segments of each day 
jour = ["00",
        "06",
        "12",
        "18"]

m = 1
d = 1


#Dataset that includes every source and its country
sourceList = "MASTER-GDELTDOMAINSBYCOUNTRY-MAY2018.txt"
sources = pd.read_csv(sourceList, sep='\t', names = ["Source", "CountryCode", 
                                                     "Country"])

countries_and_codes = sources.drop("Source", 1).drop_duplicates()

def normalise_self(data):
    # list that includes every unique country code in the data
    codes = week_data.index.unique()

    # Normalises self references.
    #
    # Records self references in "SELF" column and sets the original to 0
    # Then changes the original self reference to the mean of all other 
    # mentions of that country.
    
    for c in codes:
        try:
            data.loc[c, "SELF"] = data.loc[c, c]
            data.loc[c, c] = 0
            data.loc[c, c] = data.loc[:,c].mean()
        except:
            pass
        
def import_data(year, month, day, hour):
    # Imports the csv data and creates a dataframe including column names
    file = "{}{}{}{}.gkg.csv".format(year, month, day, hour)
    file_trans = "{}{}{}{}.translation.gkg.csv".format(year, month, day, hour)
    # Imports the untranslated files
    df = pd.read_csv(file, sep='\t', names = ["Date", 
                      "SourceCollectionIdentifier", "Source", 
                      "DocumentIdentifier", "Counts", "V2Counts", "Themes", 
                      "V2Themes", "Locations", "V2Locations", "Persons", 
                      "V2Persons", "Organizations", "V2Organizations", 
                      "V2Tone", "Dates", "GCAM", "SharingImage", 
                      "RelatedImages", "SocialImageEmbeds", 
                      "SocialVideoEmbeds", "Quotations", "AllNames", 
                      "Amounts", "TranslationInfo", "Extras"],
                      encoding = "ISO-8859-1")
    
    # Imports the translated files
    df_t = pd.read_csv(file_trans, sep='\t', names = ["Date", 
                        "SourceCollectionIdentifier", "Source", 
                        "DocumentIdentifier", "Counts", "V2Counts", "Themes", 
                        "V2Themes", "Locations", "V2Locations", "Persons", 
                        "V2Persons", "Organizations", "V2Organizations", 
                        "V2Tone", "Dates", "GCAM", "SharingImage", 
                        "RelatedImages", "SocialImageEmbeds", 
                        "SocialVideoEmbeds", "Quotations", "AllNames", 
                        "Amounts", "TranslationInfo", "Extras"],
                        encoding = "ISO-8859-1")
            
    print("{}/{}/{}/{}".format(year, month, day, hour))
    return df.append(df_t)
 
def check_timeslots(h):
    # Downloads data for 4 segments of each day for every month
    # Then it extracts each zipped file 
    # It does this for both translated and English datasets
    #
    # In rare cases, GDELT does not have a particular file available.
    # This problem will be avoided by trying the each 15 minute 
    # interval for the hour in order.
                 
    try:
        hour = h + "0000" 
        day_data = import_data(year, month, day, hour)
    except:
        try:
            hour = h + "1500"
            day_data = import_data(year, month, day, hour)
        except:
            try:
                hour = h + "3000" 
                day_data = import_data(year, month, day, hour)
            except:
                hour = h + "4500"
                day_data = import_data(year, month, day, hour)
    return day_data
       
for week in range (20):
    print("Week", week + 1)
    data_list.clear()
    for i in range(7):
        # If the day count exceeds the total days in the current month, 
        # the day counter will be reset and the month counter will go up by 1
        
        if d == monthrange(int(year), m)[1]:
            d = 1
            m += 1
            break
        
        # Makes sure that the days and months are formatted correctly
        if d < 10:
            day = "0" + str(d)
        else:
            day = str(d)
            
        if m < 10:
            month = "0" + str(m)
        else:
            month = str(m)
        
        # The algorithm checks every 15 min interval for a 
        # timeslot when data is not available.
        #
        # For robustness, this is repeated for several hours if data cannot 
        # be found.
        for h in jour:
            try:
                day_data = check_timeslots(h)
            except:
                try:
                    h = int(h) + 1 
                    if(h < 10):
                        h = "0" + str(h)
                    else:
                        h = str(h)
                    day_data = check_timeslots(h)
                except:
                    try:
                        h = int(h) + 1 
                        if(h < 10):
                            h = "0" + str(h) 
                        else:
                            h = str(h)
                        day_data = check_timeslots(h)
                    except:
                        try:
                            h = int(h) + 1 
                            if(h < 10):
                                h = "0" + str(h) 
                            else:
                                h = str(h)
                            day_data = check_timeslots(h)
                        except:
                            try:
                                h = int(h) + 1 
                                if(h < 10):
                                    h = "0" + str(h) 
                                else:
                                    h = str(h)
                                day_data = check_timeslots(h)
                            except:
                                h = int(h) + 1 
                                if(h < 10):
                                    h = "0" + str(h) 
                                else:
                                    h = str(h)
                                day_data = check_timeslots(h)
        
        # Increases the day count by 1
        d += 1
  
        #Merges the source nationality dataset with the original
        merged_data = pd.merge(day_data,sources,left_on=['Source'], 
                               right_on = ['Source'], how = 'left')
        print(merged_data["Country"].value_counts())
    
        
        def country_parser():
            # Extracts the country references from the GDELT data and creates
            # a dataframe where this data is one-hot encoded.
            cp = merged_data["Locations"].str.split(';', expand = True).apply(
                 lambda x:x.str.split('#').str[3].str[:2])

            cp = pd.get_dummies(cp, prefix = '', prefix_sep = '')
            cp = cp.groupby(by=cp.columns, axis=1).sum()
            cp[cp > 0] = 1
            
            return cp 

        countries = country_parser().fillna(0)
        
        # Creates a single dataset
        fd = merged_data[["Country", "CountryCode"]].join(countries, 
                                                         how='outer').dropna() 
        final_data = fd.groupby("CountryCode", as_index = False).mean()

        print("Dataset complete!")
        print("--- %s seconds ---" % (time.time() - start_time))
        data_list.append(final_data)
    
    # Takes the mean values from a full week and normalises it
    week_data = pd.concat(data_list).groupby("CountryCode", 
                                             as_index = False).mean()
    week_data = pd.merge(week_data, countries_and_codes, 
                         left_on=['CountryCode'], right_on = ['CountryCode'], 
                         how = 'left')
    week_data.set_index("CountryCode", inplace = True)
    normalise_self(week_data)
    
    week_data.stack()
    week_list.append(week_data)
    
# Concats each week into a single dataframe
total_data = pd.concat(week_list).fillna(0)

# Creates the window that pops up when exporting a csv
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'lightsteelblue2', 
                    relief = 'raised')
canvas1.pack()

# Exports the data as a csv file
# Allows you to name it and pick a location to save it
def exportCSV ():
    global total_data
        
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    total_data.to_csv (export_file_path, index = False)

saveAsButton_CSV = tk.Button(text='Export CSV', command=exportCSV, bg='green', 
                             fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=saveAsButton_CSV)

root.mainloop()
