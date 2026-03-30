'''
-------------------------------------------------------
Predicts values and probability for a model.
-------------------------------------------------------
Author(s): Arbert Owusu & Jeryshan Varatheswaran
Date last updated: 2026-03-29
-------------------------------------------------------
'''

from sklearn.linear_model import LinearRegression
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def predict(model, X):
    '''
    Predicts values and probability.

    Argument(s):
      model (machine learning model): training model
      X (dataframe): column / class of data

    Return(s):
      y_pred (dataframe): column / class of predicted data
      y_pred_proba: probability for y_pred
    '''
    #must add cases to handle the type of model passed so certain prediction can be pased.  
    y_pred = model.predict(X)
    if isinstance(model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor)):
      return y_pred

    y_pred_proba = model.predict_proba(X)
    return y_pred,y_pred_proba
