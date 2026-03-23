import pandas as pd
import numpy as np
from sklearn import preprocessing 


def load_dataset(file_path):
    return pd.read_csv(file_path)
#can add error handling here



#General subroutines that can be implemented in preprocessing_data function set. 
#Not sure how useful this will be.
def station_encoding(df_c):




    return df_c 



# assigns a unique integer to each Code
def transit_code_encoding(df_c):
    


    return df_c


#(N - 1, S - 2, E - 3, W-4)
def bound_encoding(df_c):


    return df_c 


#one-hot encoding of each unique Line
def line_encoding(df_c):




    return df_c



def preprocess_dataset(df):
    df = df.dropna() #drop NULL columns from df
    return df


