import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from preprocessing_data.preprocess import load_dataset, preprocess_dataset 
from models.train_model import train_model
from models.predict_model import predict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#add appropriate imports here
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

    #model = train_model(LinearRegression(), X_training_data, y_training_data)
    


    X = df.drop(columns=['Min Delay','Min Delay', 'Min Gap'])
    y = df_copy['Delay_Risk'] # yes/no classification problem
    y_reg = df_copy['Delay_Minutes'] # will have to define a formula that defines delay (ex. Time - Min Delay)
    



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    #Preprocess data here


    #Then attain features from feature_engineering module to train the model
    # train_model(model,_, _)

    df.info()

    #after all transformation and feature gathering train and predict models
    
    #model = train_model(LinearRegression(), X_training_data, y_training_data)






if __name__ == "__main__":
    main() 


