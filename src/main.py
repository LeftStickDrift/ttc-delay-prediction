import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from preprocessing_data.preprocess import load_dataset, preprocess_dataset 
from models.train_model import train_model
from models.predict_model import predict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#add appropriate imports here

#model testing purposes can move this to a seperate .py folder
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, r2_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics

def main():
    df = load_dataset('data/raw/ttc_dataset.csv')
    df = preprocess_dataset(df)
    
    model = LinearRegression()

    #attain features from feature_engineering module to train the model
    # train_model(model,_, _)

    df.info()





if __name__ == "__main__":
    main() 


