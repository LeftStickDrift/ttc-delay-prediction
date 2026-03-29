import pandas as pd
import numpy as np
#train_test_split
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV



def weekday_hour_month_encoding(df_c):
    '''
    Encodes days, hours, and months.
    Mon - Sunday (0- Mon, 6 - Sun), hour into (0 - 00:00,1- 01:00 etc... using 24 hour clock format), 
    month encoded to (1-12. 1 being January, 12 - being December)

    Argument(s):
      df_c (Pandas Dataframe): dataframe of dataset (column)

    Return(s):
      df_c (Pandas Dataframe): dataframe of dataset (column)
    '''

    day_of_week_dict = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    if df_c['Day'].dtype == "str":
        df_c["Day_Of_Week"] = df_c['Day'].str.lower().map(day_of_week_dict)

    
    
    df_c['Hour'] = pd.to_datetime(df_c['Time'], format="%H:%M").dt.hour

    df_c['Month'] = pd.to_datetime(df_c['Date']).dt.month



    return df_c


def time_group(df_c): 
    '''
    Creates a label where routes where delay occurs at x time is grouped into ranges. (delay at 2 - 2-3)

    Argument(s):
      df_c (Pandas Dataframe): dataframe of dataset (column)

    Return(s):
      df_c (Pandas Dataframe): dataframe of dataset (column)
    '''

    df_group = pd.to_datetime(df_c['Time'], format="%H:%M").dt.hour

    labels = ['late night', 'early morning', 'noon', 'afternoon', 'evening', 'night']
    bins = [0, 4, 8, 12, 16, 20, 24] 


    df_c['Time_Group'] = pd.cut(df_group, bins=bins, labels=labels, right=False)


    df_c = pd.get_dummies(df_c, columns=['Time_Group'])

    return df_c


#Creates a label for rush_hour 
def rush_hour(df_c):
    '''
    Creates a label for rush_hour

    Argument(s):
      df_c (Pandas Dataframe): dataframe of dataset (column)

    Return(s):
      df_c (Pandas Dataframe): dataframe of dataset (column)
    '''

    weekday_rush = [6,7,8,9,10]
    weekend_end_rush = [15,16,17,18,19,20]


    df_c['Rush_Hour'] = 0

    df_c.loc[df_c['Hour'].isin(weekday_rush) & df_c['Day_Of_Week'].isin([0,1,2,3,4]), 'Rush_Hour'] = 1
    df_c.loc[df_c['Hour'].isin(weekend_end_rush) & df_c['Day_Of_Week'].isin([5,6]), 'Rush_Hour'] = 1


    return df_c






