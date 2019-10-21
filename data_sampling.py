import pandas as pd


def data_sampling(df_transaction, df_identity , sampling_number):
    """"
    The shape of my original data is (590540, 394)
    For this project, we are going to use just 10% of the data 
    because of the computational costs
    df_transaction: is the original dataframe
    df_identity: identity dataframe
    sampling_prec: the number of samples
    
    this function save the output as a csv file in data folder
    """
    #Lets merege two data frames firstly
    df_identity['had_id'] = 1
    df=pd.merge(df_transaction,df_identity,how="left",on="TransactionID")
    
    print('The shape of the main data frame is:', df.shape)
    # To compute the precentage belonging to each class
    class_prec = list(df['isFraud'].value_counts(normalize = True).reset_index()['isFraud'])
    
    #The number of required samples for each class
    class_number_samples = [sampling_number - int(class_prec[1]*sampling_number), int(class_prec[1]*sampling_number)]
    
    sampled_data = []
    for class_ in [0, 1]:
        temp = df.loc[df['isFraud'] == class_].sample(class_number_samples[class_], random_state=1985)
        sampled_data.append(temp)
    IEEE_data = pd.concat(sampled_data)
    print(IEEE_data.shape)
    
    IEEE_data.to_csv('Data/Sampled_IEEE_data.csv', index = False)