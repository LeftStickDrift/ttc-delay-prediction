import pandas as pd
import numpy as np
#train_test_split
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV



#encodes days Mon - Sunday (0- Mon, 6 - Sun), hour into (0 - 00:00,1- 01:00 etc... using 24 hour clock format), month encoded to (1-12. 1 being January, 12 - being December)
def weekday_hour_month_encoding(df_c):

    #weekend label sat-sun. (0 or 1)
    return df_c


#creates a label where routes where delay occurs at x time is grouped into ranges. (delay at 2 - 2-3)
def time_group(df_c): 

    df_group = pd.to_datetime(df_c['Time']).dt.hour

    labels = ['late night', 'early morning', 'noon', 'afternoon', 'evening', 'night']
    bins = [0, 4, 8, 12, 16, 20, 24] 


    df_c['Time_Group'] = pd.cut(df_group, bins=bins, labels=labels, right=False)




    return df_c

#Function that creates labels based off hour to indicate rush-hours 
def rush_hour(df_c):

    weekday_rush = range(6,10)
    weekend_rush = range(15, 19)


    df_hour = pd.to_datetime(df_c['Time']).dt.hour

    df_c['Rush_Hour'] = 0

    df_c.loc[df_hour.isin(weekday_rush), 'Rush_Hour'] = 1
    df_c.loc[df_hour.isin(weekend_rush), 'Rush_Hour'] = 1


    return df_c






