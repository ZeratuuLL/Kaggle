import pandas as pd
from collections import Counter
import numpy as np
import os
import os.path
import csv
import json
from pandas.io.json import json_normalize
import ast
########################################################
#         rows below has to be run only once           #
########################################################

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

test = load_df(csv_path='./data/test.csv')
test['hits'] = test['hits'].apply(ast.literal_eval)

to_split=['customDimensions', 'device.device', 'geoNetwork.geoNetwork', 'totals.totals', 'hits']
# Here we extract data from columns
temp = test['customDimensions'].apply(lambda x : ast.literal_eval(x))
temp[temp.apply(lambda x: x==[])] = [{}]
temp[temp.apply(len)>0] = temp[temp.apply(len)>0].apply(lambda x: x[0])
test['customDimensions'] = temp

df_list = [test]
for column_name in to_split[:-1]:
    prefix = column_name.split('.')[0]#Get the name before the dot
    temp_df = pd.DataFrame(list(test[column_name]))
    names = temp_df.columns.values
    new_names = [prefix+'.'+str(name) for name in names]
    temp_df.rename(columns=dict(zip(names, new_names)), inplace=True)
    df_list.append(temp_df)
    
test = pd.concat(df_list, axis=1)

test.to_pickle('Test_Saved.pkl')# For later quick use

########################################################
#         rows above has to be run only once           #
########################################################

test = pd.read_pickle('Test_Saved.pkl')

test['year'] = test['date'].apply(lambda x: x//10000)
test['month'] = test['date'].apply(lambda x: (x%10000)//100)
test['month_index'] = (test['year']-2016)*12 + test['month']
test['totals.totalTransactionRevenue'] = test['totals.totalTransactionRevenue'].astype(float).fillna(0)

monthly_revenue = train.groupby(['fullVisitorId', 'year', 'month'], as_index=False).aggregate({'totals.totalTransactionRevenue':['sum'],\
                                                                                               'geoNetwork.country':['unique'],\
                                                                                               'month_index':['unique']})

monthly_revenue.columns = ['fullVisitorId', 'year', 'month', 'revenue', 'country', 'month_index']
monthly_revenue['country'] = monthly_revenue['country'].apply(lambda x:x[0])
monthly_revenue['month_index'] = monthly_revenue['month_index'].apply(lambda x:x[0])
monthly_revenue['revenue'] = monthly_revenue['revenue'].apply(lambda x:np.log(x+1))
monthly_revenue.to_pickle('Testing_Base.pkl')
