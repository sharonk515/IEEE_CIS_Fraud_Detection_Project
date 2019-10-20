#The main goal of this function is to clean and preprocess the data
import numpy as np
import pandas as pd
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


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
    cols_to_transform = [col for col in df.columns if X_train[col].dtype == object]
    
    #Getting dummies variables
    df_dummies = pd.get_dummies( X_train, columns = cols_to_transform, drop_first=True )
    
    #Return to train and test dataframes and remove train columns 
    X_train = df.loc[df['train'] == 1]
    X_train.drop(columns = 'train', axis = 1, inplace = True)
    X_test = df.loc[df['train'] == 0]
    X_test.drop(columns = 'train', axis = 1, inplace = True)
    
    return X_train, X_test

def PCA_Dimensionality_Reduction(X_train, X_test, features, n_components, plot_pca = False):
    """
    This function is used to perform PCA
    and reduce the dimmesion
    
    X_train: training data frame
    X_test: test data frame
    n_components: n_components
    features: features should be considered for pca
    """
    #Lets repalce na values first
    train_pca = X_train[features].copy()
    test_pca = X_test[features].copy()
    train_pca.fillna(train_pca.min() - 2, inplace = True)
    test_pca.fillna(train_pca.min() - 2, inplace = True)
    
    #lets define a pipline
    pipeline = Pipeline([('scaling', MinMaxScaler()), ('pca', PCA(n_components=n_components))])
    
    #Now, we train the model using train data set 
    # and transform the train data set into principle ones
    train_pca = pipeline.fit_transform(train_pca)
    
    var_explained = pipeline.named_steps['pca'].explained_variance_ratio_
    
    #convert the principle components into a data frame
    df_train_pca = pd.DataFrame(train_pca)
    
    #rename columns
    df_train_pca.rename(columns=lambda x: 'pca_v' + str(x + 1), inplace=True)
    #merge train and pca dataframes
    X_train.drop(features, axis=1, inplace = True)
    X_train.reset_index(drop = True, inplace = True)
    X_train_t = pd.concat([X_train, df_train_pca], axis=1)
    
    #transform the test dataset into prinicipal components
    test_pca = pipeline.transform(test_pca)    
        
    #Convert the principle components into a data frame
    df_test_pca = pd.DataFrame(test_pca)
    
    df_test_pca.rename(columns=lambda x: 'pca_v' + str(x + 1), inplace=True)
    #merge train and pca dataframes
    X_test.drop(features, axis=1, inplace = True)
    X_test.reset_index(drop = True, inplace = True)
    X_test_t = pd.concat([X_test, df_test_pca], axis=1)
    
     
    tot_var = sum(var_explained)
    print(f'The explained variance using {n_components} components is {round(tot_var,2)*100}%')
    
    
    if plot_pca:
        plt.figure(figsize = (10, 6))
        cum_var_exp = np.cumsum(var_explained)
        # plot explained variances
        
        plt.bar(range(1,n_components + 1), var_explained, alpha=0.5, 
                align='center', label='individual explained variance')
        plt.step(range(1,n_components + 1), cum_var_exp, where='mid',
                 label='cumulative explained variance', color = 'r')
        plt.ylabel('Explained variance ratio', fontsize=18)
        plt.xlabel('Principal component index', fontsize=18)
        plt.title('Principal Component Analysis', fontsize=22)
        plt.legend(loc='best', fontsize=16)
        
    return X_train_t, X_test_t 

def Feature_Engineering(df_train, df_test):
    """
    This function is used to generate some
    new features.
    df: input dataframe
    """
    #Building new features for emaildomain
    for feature in ['P_emaildomain', 'R_emaildomain']:
        df_train[[f'{feature}_{i}' for i in range(1, 4)]] = df_train[feature].str.split('.', expand=True)
        df_test[[f'{feature}_{i}' for i in range(1, 4)]] = df_test[feature].str.split('.', expand=True)
    
    #normalizing several columns to mean of each them for card1 and card4
    for col in ['TransactionAmt', 'D15']:
        for feature in ['card1', 'card4']:
            for measure in ['mean', 'std']: 
                df_train[f'{col}_to_{measure}_{feature}'] = df_train[f'{col}'] / df_train.\
                groupby([feature])[f'{col}'].transform(f'{measure}').copy()
                df_test[f'{col}_to_{measure}_{feature}'] = df_test[f'{col}'] / df_test.\
                groupby([feature])[f'{col}'].transform(f'{measure}').copy()

        
    #Normalizing TransactionAmt
#     df_train['TransactionAmt'] = np.log(df_train['TransactionAmt'])
#     df_test['TransactionAmt'] = np.log(df_test['TransactionAmt'])

    mean_train = df_train['TransactionAmt'].mean()
    std_train = df_train['TransactionAmt'].std()
    print(mean_train, std_train)
    df_train['TransactionAmt'] = ((df_train['TransactionAmt'] - mean_train) / std_train)
    df_test['TransactionAmt'] = ((df_test['TransactionAmt'] - mean_train) / std_train)

    return df_train, df_test


def data_cleaning_for_training(X_train, X_test):
    many_null_cols_train = [col for col in X_train.columns if X_train[col].isnull()\
                            .sum() / X_train.shape[0] > 0.9]
    many_null_cols_test = [col for col in X_test.columns if X_test[col].isnull()\
                           .sum() / X_test.shape[0] > 0.9]
    big_top_value_cols_train = [col for col in X_train.columns if X_train[col]\
                                .value_counts(dropna=False, normalize=True).values[0] > 0.9]
    big_top_value_cols_test = [col for col in X_test.columns if X_test[col].\
                               value_counts(dropna=False, normalize=True).values[0] > 0.9]
    
    cols_to_drop = list(set(many_null_cols_train + 
                            many_null_cols_test +
                            big_top_value_cols_train
                            + big_top_value_cols_test))
    
    cols_to_drop.append('TransactionID')
    cols_to_drop.append('TransactionDT')
    #print(*cols_to_drop)
    
    X_train = X_train.drop(cols_to_drop, axis=1)
    X_test = X_test.drop(cols_to_drop, axis=1)
    
    # Cleaning infinite values to NaN
    
    X_train = clean_inf_nan(X_train)
    X_test = clean_inf_nan(X_test )
    
    return X_train, X_test
    
def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)   



def data_cleaning_for_EDA(df):
    """
    The goal of this function is to clean
    the dataframe
    df: input data frame
    output is a clean data frame
    """
    # Clean R_emaildomain and P_emaildomain for just plotting
    for feature in ['R_emaildomain', 'P_emaildomain']:
        df.loc[df[feature].str.contains('yahoo', na = False), feature]  = 'Yahoo Mail'
        df.loc[df[feature].str.contains('gmail', na = False), feature] = 'Google'
        df.loc[df[feature].str.contains('hotmail|outlook|live|msn', na = False), feature] = 'Microsoft'
        df.loc[df[feature].isin(df[feature]\
                                .value_counts()[df[feature]\
                                                .value_counts() <= 200 ]\
                                     .index), feature] = "Others"
        df[feature].fillna("NoInf", inplace=True)
        
    # Clean addr1 and P addr2
    df.loc[df['addr1'].isin(df['addr1']\
                            .value_counts()[df['addr1']\
                                        .value_counts() <= 500 ]\
                             .index), 'addr1'] = "Others"
    df['addr1'].fillna("NoInf", inplace=True)
    
    df.loc[df['addr2'].isin(df['addr2']\
                        .value_counts()[df['addr2']\
                                    .value_counts() <= 10]\
                         .index), 'addr2'] = "Others"
    
    df.loc[df['C1'].isin(df['C1']\
                    .value_counts()[df['C1']\
                                .value_counts() <= 500]\
                     .index), 'C1'] = "Others"
    
    df.loc[df['C2'].isin(df['C2']\
                .value_counts()[df['C2']\
                            .value_counts() <= 350]\
                 .index), 'C2'] = "Others"
    
    return df

def remove_otliers(df, cols): 
    """
    The objective of this function is to remove outliers
    
    Here, we consider the data points beyoned mean + 3 * std
    and below mean - 3 * std as the outliers
    
    df: it is the main dataframe
    
    cols: it includes the list of columns
    to be considered for removing outliers
    
    
    it returen a series with no outliers
    
    """
    
    for col in cols:
        # calculating mean and std of the array
        data_mean, data_std = np.mean(df[col]), np.std(df[col])
        
        

    #Calculating the higher and lower cut values
    lower, upper = data_mean - 3*data_std , data_mean + 3* data_std 
    
    #removing outliers
    length_before = df.shape[0]
    df = df.loc[(df[col] < upper) & (df[col] > lower)]
    length_after = df.shape[0]

    print(f'Identified outliers for {col} is {length_before - length_after}')
    perc = round((length_before - length_after) / length_before* 100, 2) 
    print(f'The percentage of outliers for {col} is {perc}')
    return df

def reduce_mem_usage(df, verbose=True):
    """
    This function reduces the size of the dataframe
    df: the input dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary