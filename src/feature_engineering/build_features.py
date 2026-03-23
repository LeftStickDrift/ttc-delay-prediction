import pandas as pd
import numpy as np
#train_test_split
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV



#encodes days Mon - Sunday (0- Mon, 6 - Sun), hour into (0 - 00:00,1- 01:00 etc... using 24 hour clock format), month encoded to (1-12. 1 being January, 12 - being December)
def weekday_hour_month_encoding(df_c):


    return df_c


#creates a label where routes where delay occurs at x time is grouped into ranges. (delay at 2 - 2-3)
def time_group(df_c): 





    return df_c

#Function that creates labels based off hour to indicate rush-hours 
def rush_hour(df_c):




    return df_c






