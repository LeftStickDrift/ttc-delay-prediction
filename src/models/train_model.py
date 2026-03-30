from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
    #performs grid search to find best parameters
    if isinstance(model, (DecisionTreeRegressor)):
        param_grid = {
            'max_depth': list([None, 5, 10, 11, 12, 13, 14, 15]), 
            'min_samples_split': [5, 6, 7, 8, 10, 12, 14, 16], 
            'min_samples_leaf': [1, 2, 3, 4, 5]
            }
        grid = GridSearchCV(model, param_grid, cv=10, n_jobs=-1)
        grid.fit(x_train, y_train)

        print('Improved score: ', grid.best_score_)
        print('Improved parameters: ', grid.best_params_)
        
        return grid.best_estimator_
    
    elif isinstance(model, (RandomForestRegressor)):
        
        param_grid = {
            'max_depth': list([None, 5, 10, 15]), 
            'min_samples_split': [5, 10, 15, 20], 
            'min_samples_leaf': [1, 3, 5],
            'max_features': [None, 'sqrt', 'log2']
            }
        grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid.fit(x_train, y_train)

        print('Improved score: ', grid.best_score_)
        print('Improved parameters: ', grid.best_params_)

        return grid.best_estimator_
        

      
        

    model.fit(x_train, y_train)
    return model