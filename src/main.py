import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from preprocessing_data.preprocess import load_dataset, preprocess_dataset, station_encoding, transit_code_encoding, bound_encoding, line_encoding  
from feature_engineering.build_features import time_group, rush_hour, weekday_hour_month_encoding
from models.train_model import train_model
from models.predict_model import predict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import neighbors

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from visualization.visualize import get_performance_metrics, display_metrics, plot_acc

import pandas as pd


# CONSTANTS
max_depth_range_list = [0, 5, 10, 15, 20, 35, 40, 45, 50]
min_sample_split_list = [1, 2, 4, 8, 16, 32, 64]


def kfold_cv(model, X, y, k=20):
    '''
    Perform K-Fold Cross Validation given a model and x and y classes.

    Argument(s):
      model (machine learning model): model
      X (dataframe): column / class of data
      y (dataframe): column / class of data
      k (int): number of folds

    Return(s):
      model (machine learning model): K-Fold Cross Validation model
    '''

    # Check if given input is a dataframe

    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(y, pd.DataFrame):
        y = y.values
    elif isinstance(y, pd.Series):
        y = y.values
    
    # Initialize  KFold object
    kf = KFold(n_splits=k)

    # Initialize list to store the accuracy scores
    scores = []

    # Loop through the splits
    for train_index, test_index in kf.split(X):

        # Split the data into train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model on the train set
        model.fit(X_train, y_train)

        # Get the predictions for the test set
        y_pred = model.predict(X_test)

        # Compute the accuracy score
        score = accuracy_score(y_test, y_pred)

        # Append the score to the list
        scores.append(score)

    return (scores)


def grid_search(model, X, y, max_depth_range_list, min_sample_split_list):
    '''
    Perform grid search cross validation given a model, x and y classes, and maxiimum depth and mininum sample ranges.

    Argument(s):
      model (machine learning model): model
      X (dataframe): column / class of data
      y (dataframe): column / class of data
      max_depth_range_list (list): list of maximum depth values to test
      min_sample_split_list (list): list of minimum sample split values to test

    Return(s):
      best_model (machine learning model): best model from grid search
      best_params (dict): best parameters from grid search
    '''

    param_grid = {
        'max_depth': list(max_depth_range_list),
        'min_samples_split': list(min_sample_split_list)
    }
    
    grid = GridSearchCV(model, param_grid, cv=20,  n_jobs=-1)
    grid.fit(X, y)
    model.set_params(max_depth=grid.best_params_['max_depth'])

    print('Improved score: ', grid.best_score_)
    print('Improved parameters: ', grid.best_params_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return (best_model, best_params)

def main():
    df = load_dataset('data/raw/ttc_dataset.csv') #raw immutable dataset DO NOT MODIFY

    df_copy = df.copy()

    # target variables
    df_copy = df_copy.rename(columns={"Min Delay": "Delay_Minutes"})
    df_copy['Delay_Risk'] = (df_copy['Delay_Minutes'] > 15).astype(int)

    print(df_copy[['Delay_Minutes', 'Delay_Risk']].head()) 

    #preprocessing 
    df_copy = station_encoding(df_copy)
    df_copy = transit_code_encoding(df_copy)
    df_copy = bound_encoding(df_copy)
    df_copy = line_encoding(df_copy)
    df_copy = preprocess_dataset(df_copy)

    #Feature_engineering 
    df_copy = time_group(df_copy)
    df_copy = weekday_hour_month_encoding(df_copy)
    df_copy = rush_hour(df_copy)

    #remove this 
    # print(df_copy[df_copy['Delay_Minutes'] > 350]) # model is dominated by outliers
    df_copy = df_copy[df_copy['Delay_Minutes'] < 60] # remove this once you better tune your model 

    #target variables
    y_reg = df_copy['Delay_Minutes'] # regression problem
    y = df_copy['Delay_Risk'] # yes/no classification problem
    
    #predictors/features
    X = df_copy.drop(columns=['Delay_Risk','Date', 'Min Gap', 'Delay_Minutes', 'Time', 'Day', '_id'])


    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test= train_test_split(X, y, y_reg, test_size=0.2, random_state=42) # aiming to predict two targets so must split y_reg as well. 

    print(X.info())


    #Delay_minutes predictions
    print(10 * '-' + "Regression problem Results" + 10 * '-')
    print("\nLinearRegression results:")
    model = train_model(LinearRegression(), X_train, y_reg_train) # can compare this model vs RandomForest
    y_pred_lr = predict(model,X_test) # , y_pred_proba_lr
    display_metrics(y_reg_test, y_pred_lr) #visualization 

    #DecisionTree regression and grid_searchcv
    print("\nDescisionTree results:")

    model_reg_tree = train_model(DecisionTreeRegressor(random_state=42), X_train, y_reg_train)
    y_pred_tree = predict(model_reg_tree, X_test) 
    display_metrics(y_reg_test, y_pred_tree) #visualization 


    #RandomForest regression and grid_searchcv
    print("\nRandomTree results:")

    model_ran_tree = train_model(RandomForestRegressor(random_state=42, n_jobs=-1), X_train, y_reg_train)
    y_pred_rantree = predict(model_ran_tree, X_test) 
    display_metrics(y_reg_test, y_pred_rantree) #visualization 

    print("\nActual test data Statistics")
    print(y_reg_test.describe())
    
    #Delay risk predictions
    print("\n Delay Risk Predictions \n")
    print(10 * '-' + "Classification Problem Results" + 10 * '-')

    # Create several models and compare to see which would be the most accurate for our data
    
    fit_lda = train_model(LinearDiscriminantAnalysis(solver='svd'), X_train, y_train)
    y_pred_lda, y_pred_proba_lda = predict(fit_lda, X_test)   
    pm_lda = get_performance_metrics(y_test, y_pred_lda)

    fit_qda = train_model(QuadraticDiscriminantAnalysis(reg_param=0.5), X_train, y_train)
    y_pred_qda, y_pred_proba_qda = predict(fit_qda, X_test)   
    pm_qda = get_performance_metrics(y_test, y_pred_qda)
    
    fit_logit = train_model(LogisticRegression(random_state=42, max_iter=100), X_train, y_train)
    y_pred_logit, y_pred_proba_logit = predict(fit_logit, X_test)   
    pm_logit = get_performance_metrics(y_test, y_pred_logit)

    fit_dt = train_model(DecisionTreeClassifier(random_state=42), X_train, y_train)
    y_pred_dt, y_pred_proba_dt = predict(fit_dt, X_test)   
    pm_dt = get_performance_metrics(y_test, y_pred_dt)

    fit_knn = train_model(neighbors.KNeighborsClassifier(n_neighbors=5), X_train, y_train)
    y_pred_knn, y_pred_proba_knn = predict(fit_knn, X_test)   
    pm_knn = get_performance_metrics(y_test, y_pred_knn)

    result_evals = [pm_lda, pm_qda, pm_logit, pm_dt, pm_knn]
    model_names = ['LDA', 'QDA', 'Logistic Regression', 'Decision Tree', 'KNN']

    print("\n Model Performance Metrics \n")
    for i in range(len(result_evals)):
        print(f"{model_names[i]} model has an accuracy of {result_evals[i].get('Model_Accuracy'):.2f}")
    # Based on the performance metrics, the decision tree model has the highest accuracy.
    # However, since the accuracy scores are all under 80%, we need to fine tune the Decision Tree model


    # Perform K-Fold Cross Validation to further evaluate performance
    print("\n K-Fold Cross Validation Scores \n")

    kfold_scores_lda = kfold_cv(LinearDiscriminantAnalysis(solver='svd'), X, y)
    #print(kfold_scores_lda)

    kfold_scores_qda = kfold_cv(QuadraticDiscriminantAnalysis(reg_param=0.5), X, y)
    #print(kfold_scores_qda)

    kfold_scores_logit = kfold_cv(LogisticRegression(random_state=42, max_iter=100), X, y)
    #print(kfold_scores_logit)

    kfold_scores_dt = kfold_cv(DecisionTreeClassifier(random_state=42), X, y)
    #print(kfold_scores_dt)

    kfold_scores_knn = kfold_cv(neighbors.KNeighborsClassifier(n_neighbors=5), X, y)
    #print(kfold_scores_knn)

    kfold_evals = [kfold_scores_lda, kfold_scores_qda, kfold_scores_logit, kfold_scores_dt, kfold_scores_knn]

    print("\n K-Fold Cross Validation Model Performance Metrics \n")
    for i in range(len(kfold_evals)):
        print(f"{model_names[i]} model 10-Fold Cross Validation Scores\n{kfold_evals[i]}\n")
    

    # Use Grid Search to fine tune the decision tree model
    print("\n Grid Search for Decision Tree Model \n")
    best_dt_model, best_dt_params = grid_search(DecisionTreeClassifier(random_state=42), X, y, max_depth_range_list, min_sample_split_list)
    print(f"Best Decision Tree Model: {best_dt_model}")
    print(f"Best Decision Tree Parameters: {best_dt_params}")  


    # I commented this out, if you run this it will output a plot of accuracy vs. k-fold scores for DT model
    '''
    print("\n K-Fold Cross Validation Scores for Tuned Decision Tree Model \n")
    plot_acc(kfold_scores_dt)
    '''
    # Based on the results, the optimal number of kfolds seems to be 8 (based on the accuracy plot) and the
    # best decision tree model has an accuracy of 80% with a max depth of 20 and min sample split of 64 under 20 folds.

    # df_copy.info()
    # print(df_copy.head())




if __name__ == "__main__":
    main() 





