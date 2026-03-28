
#import appropriate modules
import matplotlib.pyplot as plt 
import seaborn as sns

import pandas as pd
import sklearn as skl

# Visualize Functions
# (We could/may use these for visualization for the report or within main.py)

def lrm_between_two(dataset_df, response_v, independent_v):
    '''
    Creates a linear regression equation for two factors from a dataset.

    Argument(s):
      dataset_df (Pandas Dataframe): dataframe of dataset
      response_v (str): response variable from dataset (y)
      independent_v (str): independent variable from dataset to be compared with response variable (x)

    Return(s):
      lrm_equation (str): linear regression equation
    '''

    model = skl.linear_model.LinearRegression()
    x = pd.DataFrame(dataset_df[independent_v])
    y = pd.DataFrame(dataset_df[response_v])

    model.fit(x, y)
    lrm_equation = round(model.intercept_[0], 2).astype(str) + " + (" + round(model.coef_[0][0], 2).astype(str) + " * x." + independent_v + ")"

    return(lrm_equation)


def plotScatter(dataset_df, x_col, y_col):
    '''
    Plots a scatterplot for the dataframe based on two variables.

    Argument(s):
      dataset_df (Pandas Dataframe): dataframe of dataset
      x_col (str): a column acting as independent data within dataset
      y_col (str): a column acting as dependent data within dataset

    Return(s):
      None
    '''

    dataset_df.plot(x=x_col, y=y_col, kind="scatter")

    return(None)


def plotHistogram(dataset_df, col):
    # This likely won't be useful, feel free to remove this
    '''
    Creates a histogram for a column from a dataset.

    Argument(s):
      dataset_df (Pandas Dataframe): dataframe of dataset
      col(str): column of data to plot

    Return(s):
      None
    '''

    dataset_df.hist(column=col)

    return(None)