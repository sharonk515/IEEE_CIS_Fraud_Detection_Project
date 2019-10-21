import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score, make_scorer

def Convert_LabelEncoder(X_train, X_test):
    """
    For data cleaning
    """
    #concat X_train and X_test
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
            X_train[col] = le.transform(list(X_train[col].astype(str).values))
            X_test[col] = le.transform(list(X_test[col].astype(str).values))
    return X_train, X_test


def Convert_categorical_variables(X_train, X_test):
    """
    The goal of the function to convert categorical variables to dummies
  
    """
    #concat X_train and X_test
    X_train['train'] = 1
    X_test['train'] = 0
    df = pd.concat([X_train, X_test], axis = 0)
    df.columns = X_train.columns

    cols_to_transform = [col for col in df.columns if X_train[col].dtype == object]

    #Getting dummies variables
    df_dummies = pd.get_dummies( df, columns = cols_to_transform, drop_first=True )
    
    #Return to train and test dataframes and remove train columns 
    X_train = df_dummies.loc[df['train'] == 1]
    X_train.drop(columns = 'train', axis = 1, inplace = True)
    X_test = df_dummies.loc[df['train'] == 0]
    X_test.drop(columns = 'train', axis = 1, inplace = True)
    
    return X_train, X_test
