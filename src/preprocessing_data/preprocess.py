import pandas as pd
import numpy as np
from sklearn import preprocessing 


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
#Not sure how useful this will be.
def station_encoding(df_c):
    """
    Create numeric labels for stations and return the labels and unique stations.

    Argument(s):
      df_c (pandas) - column of dataframe

    Return(s):
      unique_stations (array) - an array of tranist codes
      encoding_stations (array) - an array of integers where different integers correspond to a line
    """
    
    encoding_stations, unique_stations = pd.factorize(df_c) # Returns two arrays one encoding the column and another array containg the unique values in the column

    return (encoding_stations, unique_stations)



# assigns a unique integer to each Code
def transit_code_encoding(df_c):
    """
    Create numeric labels for transit codes and return the labels and unique transit codes.

    Argument(s):
      df_c (pandas) - column of dataframe

    Return(s):
      unique_tc (array) - an array of tranist codes
      encoding_tc (array) - an array of integers where different integers correspond to a line
    """

    encoding_tc, unique_tc = pd.factorize(df_c) # Returns two arrays one encoding the column and another array containg the unique values in the column

    return (encoding_tc, unique_tc)
    


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
    """
     Create numeric labels for line and return the labels and unique lines.

    Argument(s):
      df_c (pandas) - column of dataframe

    Return(s):
      unique_lines (array) - an array of lines
      encoding_lines (array) - an array of integers where different integers correspond to a line
    """
    
    encoding_lines, unique_lines = pd.factorize(df_c) # Returns two arrays one encoding the column and another array containg the unique values in the column

    return (encoding_lines, unique_lines)



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


