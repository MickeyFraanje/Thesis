# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:00:16 2020

@author: Mickey

Downloads the data from GDELT
"""

import requests, zipfile, io
from calendar import monthrange

# Specify which year you want to extract here:
year = "2020"

# Goes through 4 segments of each day 
jour = ["00",
        "06",
        "12",
        "18"]

def download (year, month, day, hour):
    # Downloads data from GDELT for the indicated time and date
    
    url = "http://data.gdeltproject.org/gdeltv2/{}{}{}{}.gkg.csv.zip".format(year, month, day, hour)
    url_trans = "http://data.gdeltproject.org/gdeltv2/{}{}{}{}.translation.gkg.csv.zip".format(year, month, day, hour)
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
            
    r = requests.get(url_trans, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    
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
        download(year, month, day, hour)
    except:
        try:
            hour = h + "1500"
            download(year, month, day, hour)
        except:
            try:
                hour = h + "3000" 
                download(year, month, day, hour)
            except:
                hour = h + "4500"
                download(year, month, day, hour)
    return hour
                

for m in range (0, 5):
    # Goes through every month and converts the number of the month to string
    if m < 9:
        month = "0" + str(m+1)
    else:
        month = str(m+1)
        
    # Calculates the total number of days in a given month of a given year
    days_in_month = monthrange(int(year), m+1)[1]
    
    for d in range(0, days_in_month):
        # Iterates over each day of the month
        # Goes through every day and converts the number of the day to string
        if d < 9:
            day = "0" + str(d+1)
        else:
            day = str(d+1)
        
        for h in jour:
            try:
                hour = check_timeslots(h)
            except:
                try:
                    h = int(h) + 1 
                    if(h>9):
                        h = "0" + str(h)
                    else:
                        h = str(h)
                    hour = check_timeslots(h)
                except:
                    try:
                        h = int(h) + 1 
                        if(h < 10):
                            h = "0" + str(h) 
                        else:
                            h = str(h)
                        hour = check_timeslots(h)
                    except:
                        try:
                            h = int(h) + 1 
                            if(h < 10):
                                h = "0" + str(h) 
                            else:
                                h = str(h)
                            hour = check_timeslots(h)
                        except:
                            try:
                                h = int(h) + 1 
                                if(h < 10):
                                    h = "0" + str(h) 
                                else:
                                    h = str(h)
                                hour = check_timeslots(h)
                            except:
                                h = int(h) + 1 
                                if(h < 10):
                                    h = "0" + str(h) 
                                else:
                                    h = str(h)
                                hour = check_timeslots(h)
            
            # Prints the date and time once it finishes a download to track progress
            print(year + '/' + month + '/' + day + '/' + hour)  

