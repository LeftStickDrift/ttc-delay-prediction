import numpy as np
import pandas as np 


def predict(model, X):

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    return y_pred, y_pred_proba
