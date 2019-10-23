import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, f1_score


def plot_roc_curve(X, y_obs, models, title_name, legend_names = None, colors = ['b']):
    '''
    plot Receiver operating characteristic (roc) curve
    
    Parameters
    ----------
        
        X     (pandas DataFrame) including features of data
        
        y_obs     (pandas DataFrame)  the label for observations
        
        
        models  (list) a list including the trained clf models
        
        title_name       (str)    the title for plot
        
        legend_names (list) a list including strings showing 
                            the legend for each model
                            Defult is None
        colors (list) a list including strings showing 
                            the color for each model
                            Defult is ['b']                                                
        
    Returns
    -------
        a figure including roc curve
        
    '''
    
    plt.figure(figsize = (10, 8))
    for idx, model in enumerate(models):
        logit_roc_auc = roc_auc_score(y_obs, model.predict(X))
        #fpr = false positive, #tpr = true positive
        fpr, tpr, thresholds = roc_curve(y_obs, model.predict_proba(X)[:,1])
        plt.plot(fpr, tpr, 
                 label=f'{legend_names[idx]} (area = %0.2f)' % logit_roc_auc, color = colors[idx],
                lw = 2)
#     fpr_tpr = pd.read_csv('Data/fpr_tpr.csv')
#     plt.plot(fpr_tpr['fpr'], fpr_tpr['tpr'], 
#                  label=f'Under Sampling/All data (area = 0.85)', color = '#A52A2A', lw = 2)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.title(title_name, fontsize = 20)
    plt.legend(loc='best', fontsize = 16)
    plt.savefig(f'img/{title_name}.png')


def plot_distributions_target(df_IEEE):
    '''
    plots the distributions of the targte variable ('isFraud')
    Also plots the distributions of TransactionAmt given the the targte variable ('isFraud')
    
    Parameters
    ----------        
        df_IEEE     (pandas DataFrame) input dataframe
        
    Returns
    -------
        plot described above
        
    '''
    
    df_IEEE['TransactionAmt'] = df_IEEE['TransactionAmt'].astype(float)
    total = len(df_IEEE)
    total_amt = df_IEEE.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    plt.figure(figsize=(16,6))

    plt.subplot(121)
    g = sns.countplot(x='isFraud', data=df_IEEE, )
    g.set_title("Fraud Transactions Distribution \n# 0: No Fraud | 1: Fraud #", fontsize=22)
    g.set_xlabel("Is fraud?", fontsize=18)
    g.set_ylabel('Count', fontsize=18)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=15) 

    perc_amt = (df_IEEE.groupby(['isFraud'])['TransactionAmt'].sum())
    perc_amt = perc_amt.reset_index()
    plt.subplot(122)
    g1 = sns.barplot(x='isFraud', y='TransactionAmt',  dodge=True, data=perc_amt)
    g1.set_title("% Total Amount in Transaction Amt \n# 0: No Fraud | 1: Fraud #", fontsize=22)
    g1.set_xlabel("Is fraud?", fontsize=18)
    g1.set_ylabel('Total Transaction Amount Scalar', fontsize=18)
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt * 100),
                ha="center", fontsize=15)

def transactions_distribution(df_IEEE):
    plt.figure(figsize=(16,12))
    plt.suptitle('Transaction Values Distribution', fontsize=22)
    plt.subplot(221)
    g = sns.distplot(df_IEEE[df_IEEE['TransactionAmt'] <= 1000]['TransactionAmt'])
    g.set_title("Transaction Amount Distribuition <= 1000", fontsize=18)
    g.set_xlabel("")
    g.set_ylabel("Probability", fontsize=15)

    plt.subplot(222)
    g1 = sns.distplot(np.log(df_IEEE['TransactionAmt']))
    g1.set_title("Transaction Amount (Log) Distribuition", fontsize=18)
    g1.set_xlabel("")
    g1.set_ylabel("Probability", fontsize=15)
    
    
def Distribution_feature_fraud(df, feature, rotation = 0, feature_distribution = True, loc = 'best'):
    '''
    plots distribution of the given feature ("feature") 
    and also plots distribution of it given the target label 
    
    Parameters
    ----------
        
        df     (pandas DataFrame) input dataframe
        
        feature                  (str) the name of feature for plotting
                                        required only if clf has not already been fitted 
        
        rotation                 (int)  an integer which is used to rotate xticks in plot
                                  Default: 0
                                        
        feature_distribution     (boolean) whether plots distribution of the given feature
                                  Default: True
        
        loc                       (string)  Location of the legend
                                        Default: 'best'
        
    Returns
    -------
        a plot described above
        
    '''
    # filling missing values
    df[feature].fillna('Nan', inplace = True)
    total = len(df)
    temp = pd.crosstab(df[feature], df['isFraud'], normalize='index') * 100
    temp = temp.reset_index()
    temp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    result = df[feature].value_counts()\
    .reset_index().sort_values(feature, ascending = False)['index']


    plt.figure(figsize=(14,6))
    
    if feature_distribution:
        plt.subplot(121)
        g = sns.countplot(x=feature, data=df, order = result)

        g.set_title(f"{feature} Distribution", fontsize=19)
        g.set_xlabel(f"{feature} Name", fontsize=17)
        g.set_ylabel("Count", fontsize=17)
        #g.set_ylim(0,500000)
        for p in g.patches:
            height = p.get_height()
            g.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(height/total*100),
                    ha="center", fontsize=14)
        g.set_xticklabels(g.get_xticklabels(), rotation=rotation)
        plt.subplot(122)
    g1 = sns.countplot(x=f'{feature}', hue='isFraud', data=df, order = result)
    plt.legend(title='Fraud', loc=loc, labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=f'{feature}', y='Fraud', data=temp,
                       color='black', order = result, legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)
    

    g1.set_title(f"{feature} by Target (isFraud)", fontsize=19)
    g1.set_xlabel(f"{feature} Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    g1.set_xticklabels(g1.get_xticklabels(), rotation=rotation)
    
def plot_feature_importances(clf, X_train, y_train=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator
    
    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it
        
        X_train     (pandas DataFrame)
        
        y_train     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted 
        
        top_n       (int)               Plot the top_n most-important features
                                        Default: 10
                                        
        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)
        
        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False
        
    Returns
    -------
        the pandas dataframe with the features and their importance
        
    '''
    
    __name__ = "plot_feature_importances"           
            
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.ylabel('')
    plt.xlabel('Feature Importance Score ', fontsize = 16)
    plt.yticks(fontsize = 18)
    plt.title('Feature Importance Score using Random Forest', fontsize = 18)
    plt.savefig(f'img/feature_importance.png', bbox_inches = "tight")
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp