
#train-test split 
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV


def train_model(model, x_train, y_train):
    '''
    Trains model given x and y classes.

    Argument(s):
      model (machine learning model): model
      x_train (dataframe): column / class of data
      y_train (dataframe): column / class of data

    Return(s):
      model (machine learning model): training model
    '''

    model.fit(x_train, y_train)
    return model