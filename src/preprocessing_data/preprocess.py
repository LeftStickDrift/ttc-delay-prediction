'''
-------------------------------------------------------
Functions used to proprocess the dataset.
-------------------------------------------------------
Author(s): Arbert Owusu & Jeryshan Varatheswaran
Date last updated: 2026-03-29
-------------------------------------------------------
'''

import pandas as pd


def load_dataset(file_path):
    """
    Load a dataset from a given file/destination.

    Argument(s):
      file_path (string) - file path / origin

    Return(s):
      (dataframe / None) - dataframe of the dataset or None if invalid
    """

    if (isinstance(pd.read_csv(file_path), pd.DataFrame)):
        return pd.read_csv(file_path) # Dataset
    
    else:
        return None # returns None if invalid file_path is passed



#General subroutines that can be implemented in preprocessing_data function set.  
def station_encoding(df_c):
    """
    Create numeric labels for stations and return the labels and unique stations.

    Argument(s):
      df_c (pandas) - column of dataframe

    Return(s):
      df_c(panda) - modified Station column with unique numerical values
    """
    
    df_c['Station'] = pd.factorize(df_c['Station'])[0] # 

    return df_c



# assigns a unique integer to each Code
def transit_code_encoding(df_c):
    """
    Create numeric labels for transit codes and return the labels and unique transit codes.

    Argument(s):
      df_c (pandas) - column of dataframe

    Return(s):
      df_c(panda) - modified Station column with unique numerical values
    """

    df_c['Code'] = pd.factorize(df_c['Code'])[0] # 

    return df_c
    


#(N - 1, S - 2, E - 3, W-4)
def bound_encoding(df_c):
    """
    Create numeric labels for bound direction and return the labels.

    Argument(s):
      df_c (string) - data within column
      col(str): column of data to plot

    Return(s):
      (int) - corresponding integer
    """

    
    df_c['Bound'] = pd.factorize(df_c['Bound'])[0]
    
    
    
    return df_c



#one-hot encoding of each unique Line
def line_encoding(df_c):
    """
     Create numeric labels for line and return the labels and unique lines.

    Argument(s):
      df_c (pandas) - column of dataframe

    Return(s):
      df_c(panda) - modified Line column with unique numerical values
    """
    
    df_c['Line'] = pd.factorize(df_c['Line'])[0] #

    return df_c



def preprocess_dataset(df):
    """
     Preprocesses dataset.

    Argument(s):
      df (dataframe) - dataframe

    Return(s):
      df (dataframe) - processed dataframe
    """

    df = df.dropna() #drop NULL columns from df
    return df


