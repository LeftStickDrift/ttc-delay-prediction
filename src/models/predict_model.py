import numpy as np
import pandas as np 


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
    # y_pred_proba = model.predict_proba(X)
    return y_pred  #,y_pred_proba
