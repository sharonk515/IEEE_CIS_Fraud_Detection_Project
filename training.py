import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def sampling_data(X_train, y_train, sampling_method):
    """
    This function gets the X_train and y_train and
    resample them using the following methods
    1. Oversampling
    2. Undersampling
    2. SMOTE
    """
    df_s = X_train.reset_index(drop = True).copy()
    df_s['target'] = y_train.reset_index(drop = True)
    if sampling_method == 'Oversampling':
        # Oversampling minority
        oversampled = resample(df_s.loc[df_s['target'] == 1],
                               replace=True, # sample with replacement
                               n_samples=sum(df_s['target'] == 0), # match number in majority class
                               random_state=1985) # reproducible results
        output_df = pd.concat([df_s.loc[df_s['target'] == 0], oversampled])
        X_train_o, y_train_o = output_df.drop(columns = 'target', axis = 1), output_df['target']
    elif sampling_method == 'Undersampling':
        # Oversampling minority
        undersampled = resample(df_s.loc[df_s['target'] == 0],
                               replace=False, # sample without replacement
                               n_samples=sum(df_s['target'] == 1), # match number in majority class
                               random_state=1985) # reproducible results
        output_df = pd.concat([df_s.loc[df_s['target'] == 1], undersampled])
        X_train_o, y_train_o = output_df.drop(columns = 'target', axis = 1), output_df['target']
    elif sampling_method == 'SMOTE':
        sm_model = SMOTE(random_state=1985, ratio=1.0)
        cols = [col for col in df_s.columns if col != 'target']
        X_train_o, y_train_o = sm_model.fit_sample(df_s[cols], df_s['target'])

    return X_train_o, y_train_o

def random_forest_param_selection(X_train, X_test, y_train, y_test, nfolds, n_jobs = None):
    """
    Thsi function is written to perform RF and tune hyperparameters
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    params = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

    # estimator
#     pipe = Pipeline([
#         ('SC',StandardScaler()),
#         ('RF',RandomForestClassifier())
#          ])
    skf = StratifiedKFold(n_splits=nfolds, 
                          shuffle=True, 
                          random_state=1985)
    
    grid_search = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                     param_distributions = params,
                                     cv = skf, 
                                     scoring = 'roc_auc',
                                     n_jobs = n_jobs).fit(X_train, y_train)
    print('The training roc_auc_score is:', round(grid_search.best_score_, 3))
    print('The best parameters are:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_auc_roc = round(roc_auc_score(y_test, y_pred), 2)
    print('The test roc_auc_score is:', test_auc_roc)
    return best_model, y_pred

def logistic_regression_param_selection(X_train, X_test, y_train, y_test, nfolds, n_jobs = None):
    """
    Thsi function is written to perform logistic regression and tune hyperparameters
    """
    # Inverse of regularization strength; must be 
    # a positive float. Like in support vector machines,
    # smaller values specify stronger regularization.
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
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
    print('The test roc_auc_score is:', test_auc_roc)
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
