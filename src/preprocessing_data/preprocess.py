import pandas as pd
import numpy as np
from sklearn import preprocessing 


def load_dataset(file_path):

    if (isinstance(pd.read_csv(file_path), pd.DataFrame)):
        return pd.read_csv(file_path) # Dataset
    
    else:
        return None # returns None if invalid file_path is passed



#General subroutines that can be implemented in preprocessing_data function set. 
#Not sure how useful this will be.
def station_encoding(df_c):




    return df_c 



# assigns a unique integer to each Code
def transit_code_encoding(df_c):
    


    return df_c


#(N - 1, S - 2, E - 3, W-4)
def bound_encoding(df_c):
    """Create numeric labels for bound direction and return the label."""
    if(df_c == 'N'):
        return 1
    elif(df_c == 'S'):
        return 2
    elif(df_c == 'E'):
        return 3
    elif(df_c == 'W'):
        return 4



#one-hot encoding of each unique Line
def line_encoding(df_c):
    """Create numeric labels for line and return the label."""
    #We will need to find a way to encode lines, transits, and station using some sort of loop or built-in function
   

    return df_c



def preprocess_dataset(df):
    df = df.dropna() #drop NULL columns from df
    return df


