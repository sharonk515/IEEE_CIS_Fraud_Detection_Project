import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_distributions_target(df_IEEE):
    """
    This function plots the distributions of the targte variable
    """
    
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

    plt.figure(figsize=(16,12))
    
    
def Distribution_feature_fraud(df, feature, rotation = 0, feature_distribution = True, loc = 'best'):
    """
    This function is written to plot the distruibution 
    of the "feature" and also the distribution of fraud 
    based on this feature
    """
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