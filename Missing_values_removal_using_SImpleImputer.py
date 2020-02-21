import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer

def Missing_values_removal(feature_df,dataset_df):
    strategy=''
    if(feature_df.dtype=='object'):
        strategy='most_frequent'
    else:
        strategy='mean'
    imputer=SimpleImputer(strategy=strategy)
    transform_feature_df=pd.DataFrame(imputer.fit_transform(feature_df.values.reshape(-1,1)))
    dataset_df[feature_df.name]=transform_feature_df.values
    return dataset_df

def features(filename):
    dataset=pd.read_csv(filename)
    cols=dataset.columns
    for i in cols:
            dataset=Missing_values_removal(dataset[i],dataset)      
    return 

def main():
    arguments=sys.argv[1:]
    if(len(arguments)!=1):
        print('Usage python output.py <InputDataFile.py>')
        sys.exit(1)
    filename=arguments[0]
    mod_dataset=features(filename)
    mod_dataset.to_csv('new_'+filename,index=False)

if __name__=='__main__':
    main()