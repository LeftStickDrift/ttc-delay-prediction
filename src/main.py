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

from visualization.visualize import get_performance_metrics, display_metrics


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

    # Based on results above, select model that seemingly has the best data (or possibly fine tune model further) and
    # then conduct visualization of results

    # df_copy.info()
    # print(df_copy.head())




if __name__ == "__main__":
    main() 


