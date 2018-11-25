import os
import os.path
import json
import pandas as pd
from pandas.io.json import json_normalize
import csv
import ast
import time
import argparse

parser = argparse.ArgumentParser(description='Parameters for Continuous Control')
parser.add_argument('-n', '--nrows', type=int,help='Input how many rows you want to read in')
args = parser.parse_args()

nrows =  args.nrows

#Set maximal read in rows 
csv.field_size_limit(2147483647)

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

t1=time.time()
if nrows>0:
    train=load_df(csv_path='./data/train.csv', nrows=nrows)
else:
    train=load_df(csv_path='./data/train.csv')
print('Loading Finished! Time usage: {}'.format(time.time()-t1))

to_split=['customDimensions', 'device.device', 'geoNetwork.geoNetwork', 'totals.totals', 'hits']

# Here we extract data from columns
temp = train['customDimensions'].apply(lambda x : ast.literal_eval(x))
temp[temp.apply(lambda x: x==[])] = [{}]
temp[temp.apply(len)>0] = temp[temp.apply(len)>0].apply(lambda x: x[0])
train['customDimensions'] = temp

df_list = [train]
for column_name in to_split[:-1]:
    prefix = column_name.split('.')[0]#Get the name before the dot
    temp_df = pd.DataFrame(list(train[column_name]))
    names = temp_df.columns.values
    new_names = [prefix+'.'+str(name) for name in names]
    temp_df.rename(columns=dict(zip(names, new_names)), inplace=True)
    df_list.append(temp_df)
    
df = pd.concat(df_list, axis=1)
df.iloc[:10000,].to_pickle('Train_Saved_10000.pkl')
df.iloc[:100000,].to_pickle('Train_Saved_100000.pkl')
df.iloc[:500000,].to_pickle('Train_Saved_500000.pkl')
df.iloc[:,12:26].to_pickle('trafficSource.pkl')
df.iloc[:,26:28].to_pickle('CustomDimensions.pkl')
df.iloc[:,28:44].to_pickle('device.pkl')
df.iloc[:,44:55].to_pickle('geoNetwork.pkl')
df.iloc[:,55:65].to_pickle('totals.pkl')
df.to_pickle('Train_Saved.pkl')