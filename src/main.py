import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from preprocessing_data.preprocess import load_dataset, preprocess_dataset, station_encoding, transit_code_encoding, bound_encoding, line_encoding  
from feature_engineering.build_features import time_group, rush_hour, weekday_hour_month_encoding
from models.train_model import train_model
from models.predict_model import predict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#model testing purposes can move this to a seperate .py folder
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, r2_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics

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

    #model = train_model(LinearRegression(), X_training_data, y_training_data)

    #target variables
    y_reg = df_copy['Delay_Minutes'] # regression problem
    y = df_copy['Delay_Risk'] # yes/no classification problem
    
    #predictors/features
    X = df_copy.drop(columns=['Delay_Risk','Date', 'Min Gap', 'Delay_Minutes', 'Time', 'Day', '_id'])



    # test_cp = time_group(df_copy)
    # print(test_cp['Time_Group'])

    

    # test3_cp = weekday_hour_month_encoding(df_copy)
    # print(test3_cp['Day_Of_Week'] == 1)
    # print(test3_cp['Hour'].head())
    # print(test3_cp['Month'].min())
    # test2_cp = rush_hour(df_copy)
    # print(test2_cp[test2_cp['Rush_Hour'] == 0])


    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test= train_test_split(X, y, y_reg, test_size=0.2, random_state=42) # aiming to predict two targets so must split y_reg as well. 

    print(X.info())


    #Preprocess data here



    #Then attain features from feature_engineering module to train the model
    # train_model(model,_, _)

    # df_copy.info()
    # print(df_copy.head())

    #after all transformation and feature gathering train and predict models
    
    #model = train_model(LinearRegression(), X_training_data, y_training_data)
    




if __name__ == "__main__":
    main() 


