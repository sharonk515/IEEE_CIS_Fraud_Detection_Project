import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    
    #Columns to be converted to dummies
    #columns_with_large_numbers of unqiue values
    col_large_unqiues = ['id_13','id_14','id_17','id_18',
                         'id_19','id_20','id_21','id_25',
                         'id_26','id_30',
                         'id_31','id_33']
    #cols_to_transform = [col for col in df.columns if X_train[col].dtype == object]
    cats = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 
            'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 
            'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 
            'id_27', 'id_28', 'id_29','id_30', 'id_31', 
            'id_32', 'id_33', 'id_34', 'id_35', 'id_36',
            'id_37', 'id_38', 'DeviceType', 'DeviceInfo',
            'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 
            'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 
            'M7', 'M8', 'M9','P_emaildomain_1', 'P_emaildomain_2',
            'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 
            'R_emaildomain_3',  'DeviceType', 'DeviceInfo', 'device_name', 
            'device_version', 'OS_id_30', 'version_id_30', 'browser_id_31',
            'version_id_31', 'screen_width', 'screen_height', 'had_id']
    cols_to_transform = [col for col in cats if col in X_train.columns]
    cols_to_transform = [col for col in cats if col not in col_large_unqiues]
    
    #Getting dummies variables
    df_dummies = pd.get_dummies( df, columns = cols_to_transform, drop_first=True )
    
    #Return to train and test dataframes and remove train columns 
    X_train = df_dummies.loc[df['train'] == 1]
    X_train.drop(columns = 'train', axis = 1, inplace = True)
    X_test = df_dummies.loc[df['train'] == 0]
    X_test.drop(columns = 'train', axis = 1, inplace = True)
    
    return X_train, X_test
