import pandas as pd
from collections import Counter

train = pd.read_pickle('Train_Saved.pkl')

############################# totals #############################
train['totals.totalTransactionRevenue'] = train['totals.totalTransactionRevenue'].astype(float).fillna(0).apply(lambda x: np.log(x+1))

train['totals.haveRevenue'] = train['totals.totalTransactionRevenue']>0

# relationship between number of pages and rate of purchase
train['totals.pageviews'] = train['totals.pageviews'].astype(float)

# relationship between geo information and shopping rate
country_rate = train.groupby(['geoNetwork.country'])['totals.haveRevenue'].mean()
continent_rate = train.groupby(['geoNetwork.continent'])['totals.haveRevenue'].mean()
subcontinent_rate = train.groupby(['geoNetwork.subContinent'])['totals.haveRevenue'].mean()

# relationship between geo information and shopping revenue
country_value = train.groupby(['geoNetwork.country'])['totals.totalTransactionRevenue'].mean()
continent_value = train.groupby(['geoNetwork.continent'])['totals.totalTransactionRevenue'].mean()
subcontinent_value = train.groupby(['geoNetwork.subContinent'])['totals.totalTransactionRevenue'].mean()

#############################  hits  ##############################
def check_inside(x, ind):
    '''
    check whether 'contentGroup' i a simplified version of 'pagePathLevel'
    '''
    ind = str(ind)
    result = 0
    key1 = 'contentGroup'+ind
    key2 = 'pagePathLevel'+ind
    for dicts in x:
        if dicts['contentGroup'][key1]!='(not set)':
            if dicts['contentGroup'][key1].lower() not in dicts['page'][key2]:
                result += 1
    return result

def get_content(x):
    '''
    Get all values for contentGroup variable, return the values as a list
    '''
    result = [[], [], []]
    for dicts in x:
        temp_dict = dicts['contentGroup']
        result[0].append(temp_dict['contentGroup1'])
        result[1].append(temp_dict['contentGroup2'])
        result[2].append(temp_dict['contentGroup3'])
    return result

t1 = time.time()
contents = train['hits'].apply(get_content)
content1 = contents.apply(lambda x: x[0])
content2 = contents.apply(lambda x: x[1])
content3 = contents.apply(lambda x: x[2])
content1_counter = content1.apply(Counter)
content2_counter = content2.apply(Counter)
content3_counter = content3.apply(Counter)
unique_content1 = list(content1_counter.sum().keys()) #['(not set)', 'Google', 'YouTube', 'Android']
unique_content2 = list(content2_counter.sum().keys()) #['Bags', '(not set)', 'Apparel', 'Brands', 'Electronics', 'Drinkware', 'Accessories', 'Nest', 'Office', 'Lifestyle']
unique_content3 = list(content3_counter.sum().keys()) #['(not set)', 'Womens', 'Mens']
time.time()-t1 # 78s

t1 = time.time()
content1_count = content1_counter.apply(lambda x: [x[key] for key in unique_content1])
content2_count = content2_counter.apply(lambda x: [x[key] for key in unique_content2])
content3_count = content3_counter.apply(lambda x: [x[key] for key in unique_content3])
time.time()-t1

train['content1_count'] = content1_count
train['content2_count'] = content2_count
train['content3_count'] = content3_count

temp_table = pd.DataFrame({'fullVisitorId':train['fullVisitorId'].values.tolist()})
for i in range(len(unique_content1)):
    temp_table['content1.'+unique_content1[i]] = content1_count.apply(lambda x: x[i])
for i in range(len(unique_content2)):
    temp_table['content1.'+unique_content2[i]] = content2_count.apply(lambda x: x[i])
for i in range(len(unique_content3)):
    temp_table['content1.'+unique_content3[i]] = content3_count.apply(lambda x: x[i])

temp_table = temp_table.groupby(['fullVisitorId']).sum()
alphas = temp_table.iloc[:,1:].sum()+1

############################ geoNetwork columns  ##############################
continent = train['geoNetwork.continent']
continent_counter = continent.apply(lambda x: Counter([x]))
unique_continent = list(continent_counter.sum().keys())
continent_count = continent_counter.apply(lambda x: [x[key] for key in unique_continent])

for i in range(len(unique_continent)):
    temp_table['continent.'+unique_continent[i]] = continent_count.apply(lambda x:x[i])

'''
or use the function below
'''
    
    
def geo_count(x):
    y='geoNetwork.'+x
    temp = train[y]
    temp_table = train['fullVisitorId'].to_frame()
    temp_counter = temp.apply(lambda x: Counter([x]))
    unique_temp = list(temp_counter.sum().keys())
    temp_count = temp_counter.apply(lambda x: [x[key] for key in unique_temp])
    for i in range(len(unique_temp)):
        temp_table[x+'.'+unique_temp[i]] = temp_count.apply(lambda x:x[i])
    return temp_table

############################ others  ##############################
def get_year(x):
    return x//10000
def get_month(x):
    return (x%10000)//100
train['year'] = train['date'].apply(get_year)
train['month'] = train['date'].apply(get_month)

monthly_revenue = train.groupby(['fullVisitorId', 'year', 'month'])['totals.totalTransactionRevenue'].sum()

###################### remove unless columns ######################
drop_list = []

###################### bayesian calculation blablabla ######################
def dir_alpha(x):
    '''
    x is a data frame containing all rows, first column is fullvistorID
    alpha_i is equal to N_i(counts for all visitors in category i)+1
    return a pandas dataframe with one row
    '''
    return (x.iloc[:,1:].sum()+1).to_frame().T

def series_to_frame(x):
    return x.to_frame()
