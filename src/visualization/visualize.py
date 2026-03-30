
#import appropriate modules
import matplotlib.pyplot as plt 
import seaborn as sns

import numpy as np

import sklearn as skl
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Visualize Functions
# (We could/may use these for visualization for the report or within main.py)

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


def plot_tree(tree): # Keep function (remove this comment later)
    '''
    Plots a decision tree.
    
    Args:
        tree (decision tree): a decisiion tree model

    Return:
        None
    '''
    """Plots a decision tree. Returns: None"""
    plt.figure(figsize=(15,10))
    tree.plot_tree(tree, filled=True)
    plt.show()


def plot_scatterplot_model(model_name, x_col_name, y_col_name,X_test, y_pred):
    '''
    Plots a scatterplot of a given model.
    
    Args:
        model (machine learning model): a model
        model_name (string): name of the model
        X_test (dataframe): a dataframe containing the test data for the model
        y_pred (array): a dataframe column class representing predicted values (model predictions)

    Return:
        None
    '''

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test[x_col_name], y=X_test[y_col_name], hue=y_pred)

    plt.title(model_name + ' Predictions')
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.show()


def plot_actual_and_predicted(model_name,y_test, y_pred):
    '''
    Plots a the actual data vs the predicted data of a given model.
    
    Args:
        y_test (dataframe): a dataframe column containing the actual response
        y_pred (array): a dataframe column class representing predicted values (model predictions)

    Return:
        None
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(x=y_test, y=y_pred)
    a,b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, a* y_test + b, color="black") #line of best fit
    plt.title(f'{model_name} Actual values vs ' + 'Predicted values')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()


