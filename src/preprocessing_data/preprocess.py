import pandas as pd
import numpy as np
from sklearn import preprocessing 


def load_dataset(file_path):
    return pd.read_csv(file_path)
#can add error handling here


def preprocess_dataset(df):
    df = df.dropna() #drop NULL columns from df
    return df


