import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
import gc

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score

def logistic_regression_param_selection(X_train, X_test, y_train, y_test, nfolds, n_jobs = None):
    """
    Thsi function is written to perform logistic regression and tune hyperparameters
    """
    # Inverse of regularization strength; must be 
    # a positive float. Like in support vector machines,
    # smaller values specify stronger regularization.
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # Used to specify the norm used in the penalization
    penalty = ['l1', 'l2']
    params = {'LG__C': C, 'LG__penalty': penalty}
    # estimator
    pipe = Pipeline([
        ('SC',StandardScaler()),
        ('LG',LogisticRegression())
         ])
    skf = StratifiedKFold(n_splits=nfolds, 
                          shuffle=True, 
                          random_state=1985)
    
    grid_search = RandomizedSearchCV(estimator = pipe,
                                     param_distributions = params,
                                     cv = skf, 
                                     scoring = 'roc_auc',
                                     n_jobs = n_jobs).fit(X_train, y_train)
    print('The training roc_auc_score is:', round(grid_search.best_score_, 3))
    print('The best parameters are:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_auc_roc = round(roc_auc_score(y_test, y_pred), 2)
    print('The training roc_auc_score is:', test_auc_roc)
    return best_model, y_pred

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
