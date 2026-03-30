
#import appropriate modules
import matplotlib.pyplot as plt 
import seaborn as sns

import numpy as np
import pandas as pd

import sklearn as skl
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Visualize Functions
# (We could/may use these for visualization for the report or within main.py)

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


def display_metrics(y_true, y_pred): 
    '''
    Display the mean square error, root mean square error, and R^2 metric
    about the model.
    
    Args:
        y_true  (array): actual data
        y_predicted (array): predicted data

    Return:
        None
    '''
    
    print(f'Mean Square Error: {skl.metrics.mean_squared_error(y_true, y_pred)}')
    print(f'Root Mean Square Error: {np.sqrt(skl.metrics.mean_squared_error(y_true, y_pred))}')
    print(f'R-Square: {skl.metrics.r2_score(y_true, y_pred)}')


def get_performance_metrics(y_test, y_predicted): # Keep function (remove this comment later)
    '''
    Calculate accuracy, precision, recall, f1-score, and kappa score. Returns: Dictionary of parameters
    
    Args:
        y_test  (dataframe): a dataframe column of a class representing test values
        y_predicted (array): a dataframe column of the same class representing predicted values (model predictions)

    Return:
        (dict): dictionary containing performance metrics
    '''
    
    model_accuracy = accuracy_score(y_test, y_predicted)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_test, y_predicted, zero_division = 0)
    model_kappa = metrics.cohen_kappa_score(y_test, y_predicted)

    # Confusion matrix
    model_confusion_matrix = confusion_matrix(y_test, y_predicted)

    # Return as dictionary
    return {'Model_Accuracy': model_accuracy, 'Model_Precision': model_precision, 'Model_Recall': model_recall, 'Model_F1_Score': model_f1, 'Model_Kappa': model_kappa, 'Confusion_Matrix': model_confusion_matrix}


def plot_acc(scores):
    """Creates a plot of accuracy vs k-fold. Returns: None"""
    '''
    Creates a plot of accuracy vs k-fold.
    
    Args:
        scores (list): a list of accuracy scores for each fold

    Return:
        None
    '''

    plt.plot(scores)

    plt.xlabel("Fold")
    plt.ylabel("Accuracy Score")
    plt.title("K-Fold Cross Validation")
    #plt.draw()
    plt.tight_layout()
    plt.show()
    plt.close()