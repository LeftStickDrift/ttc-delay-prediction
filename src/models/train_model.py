
#train-test split 
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV


def train_model(model, x_train, y_train):

    model.fit(x_train, y_train)
    return model


