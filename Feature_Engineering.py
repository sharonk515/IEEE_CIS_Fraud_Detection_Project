import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import seaborn as sns


def PCA_Dimensionality_Reduction(df_train, df_test, features, n_components, plot_pca = False):
    """
    This function is used to perform PCA
    and reduce the dimmesion
    
    df_train: training data frame
    df_test: test data frame
    n_components: n_components
    features: features should be considered for pca
    """
    #lets normalize the data before doing PCA
    for feature in features:
        min_2 = df_train[feature].min() - 2
        df_train[feature].fillna(min_2, inplace = True)
        df_test[feature].fillna(min_2, inplace = True)
        df_train[feature] = minmax_scale(df_train[feature], feature_range=(0,1))
        df_test[feature] = minmax_scale(df_test[feature], feature_range=(0,1))
      
    pca_model = PCA(n_components=n_components, random_state=1364)
    principal_Components = pca_model.fit_transform(df_train[features])
    
    tot_var = sum(pca_model.explained_variance_ratio_)
    print(f'The explained variance using {n_components} components is {round(tot_var,2)*100}%')
    
    df_train_pca = pd.DataFrame(principal_Components)
    
    df_train_pca.rename(columns=lambda x: 'pca_v' + str(x + 1), inplace=True)
    
    df_train = pd.concat([df_train.drop(features, axis=1), df_train_pca], axis=1)
    
    df_test_pca = pd.DataFrame(pca_model.transform(df_test[features]))  
    
    df_test_pca.rename(columns=lambda x: 'pca_v' + str(x + 1), inplace=True)
    
    df_test = pd.concat([df_test.drop(features, axis=1), df_test_pca], axis=1)
    
    if plot_pca:
        plt.figure(figsize = (10, 6))
        cum_var_exp = np.cumsum(pca_model.explained_variance_ratio_)
        # plot explained variances
        
        plt.bar(range(1,n_components + 1), pca_model.explained_variance_ratio_, alpha=0.5, 
                align='center', label='individual explained variance')
        plt.step(range(1,n_components + 1), cum_var_exp, where='mid',
                 label='cumulative explained variance', color = 'r')
        plt.ylabel('Explained variance ratio', fontsize=18)
        plt.xlabel('Principal component index', fontsize=18)
        plt.title('Principal Component Analysis', fontsize=20)
        plt.legend(loc='best')
        
    return df_train, df_test 

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
    df_train['TransactionAmt'] = np.log(df_train['TransactionAmt'])
    df_test['TransactionAmt'] = np.log(df_test['TransactionAmt'])

    mean_train = df_train['TransactionAmt'].mean()
    std_train = df_train['TransactionAmt'].std()
    print(mean_train, std_train)
    df_train['TransactionAmt'] = ((df_train['TransactionAmt'] - mean_train) / std_train)
    df_test['TransactionAmt'] = ((df_test['TransactionAmt'] - mean_train) / std_train)

    return df_train, df_test
