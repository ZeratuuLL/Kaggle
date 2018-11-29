import pandas as pd
from collections import Counter
import numpy as np

train = pd.read_pickle('Train_Saved.pkl')

############################# totals #############################
train['totals.totalTransactionRevenue'] = train['totals.totalTransactionRevenue'].astype(float).fillna(0).apply(lambda x: np.log(x+1))

train['totals.haveRevenue'] = train['totals.totalTransactionRevenue']>0

# relationship between geo information and shopping rate, this is going to be modeled by binomial
# relationship between geo information and shopping revenue, this is going to be modeled by normal-inverse gamma distribution

country_prior = train.groupby(['geoNetwork.country']).aggregate({'totals.totalTransactionRevenue':['mean','count'],\
                                                                 'totals.haveRevenue':['sum']})
country_prior.columns=['mu','n','raten']
country_prior.to_pickle('country_prior.pkl')

nu=train.shape[0]
mu0=train['totals.totalTransactionRevenue'].mean()    
alpha = train['totals.haveRevenue'].sum()+1
beta = train.shape[0]+2-alpha
country_parameters=np.array([nu, mu0, alpha, beta])
np.save('country_parameters.npy',country_parameters)

c=0.01
country_posterior = country_prior.copy()
country_posterior['country_param1'] = (c*nu*mu0+country_prior['n']*country_prior['mu'])/(c*nu+country_prior['n'])
country_posterior['country_param2'] = (c*alpha+country_prior['raten'])/(c*alpha+c*beta+country_prior['raten'])
country_posterior = country_posterior.iloc[:,-2:]
country_posterior.to_pickle('country_posterior.pkl')
                       
# page visit number
#construct a data frame with 'fullVistorId' as index and two coulmns: 'pageview': sum of pagevews, 'counts': number of same person. 

temp = pd.DataFrame({'fullVisitorId':train['fullVisitorId'].values.tolist()})
temp['counts'] = 1
temp['pageview'] = train['totals.pageviews'].astype(float).fillna(1)
temp['pageview'] = np.log(temp['pageview'])
temp_table = temp.groupby(['fullVisitorId']).aggregate({'counts':['count'],  'pageview':['sum']})
temp_table.columns = ['counts','pageview']   
temp_table.to_pickle('pageView_prior.pkl')
                       
#get prior \alpha_0, \beta_0
(alpha_0, beta_0) = (temp_table['counts'].sum()+1, temp_table['pageview'].sum()) 
np.save('pageView.npy',np.array([alpha_0, beta_0]))
                       
#get posterier \alpha, \beta for each user
gamma = 0.00001  #gamma  weights parameter
temp_table['beta'] = gamma*beta_0 + temp_table['pageview']  
temp_table['alpha'] = gamma*alpha_0 + temp_table['counts']  

#get posterior mean for each user
temp_table['mean'] =  temp_table['beta']/(temp_table['alpha']-1)    #     \alpha>1            
                       
                       
#############################  hits  ##############################
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

# get counts for content for each row as well as the total count for each content
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

# transform Counter to list
content1_count = content1_counter.apply(lambda x: [x[key] for key in unique_content1])
content2_count = content2_counter.apply(lambda x: [x[key] for key in unique_content2])
content3_count = content3_counter.apply(lambda x: [x[key] for key in unique_content3])

# add into DataFrame
train['content1_count'] = content1_count
train['content2_count'] = content2_count
train['content3_count'] = content3_count
train.iloc[:,-3:].to_pickle('Train_content_count.pkl')
train = train.iloc[:,:-3]

# calculate prior
temp_table = pd.DataFrame({'fullVisitorId':train['fullVisitorId'].values.tolist()})
for i in range(len(unique_content1)):
    temp_table['content1.'+unique_content1[i]] = content1_count.apply(lambda x: x[i])
for i in range(len(unique_content2)):
    temp_table['content2.'+unique_content2[i]] = content2_count.apply(lambda x: x[i])
for i in range(len(unique_content3)):
    temp_table['content3.'+unique_content3[i]] = content3_count.apply(lambda x: x[i])

del content1_count, content2_count, content3_count, unique_content1, unique_content2, unique_content3
                       
# get total content counts for each user
temp_table = temp_table.groupby(['fullVisitorId']).sum()#After this fullVisitorId becomes Index
temp_table.to_pickle('content_prior.pkl')
alphas = temp_table.sum()+1 #This is the prior parameters for multinomial 
alphas.to_pickle('content_alphas.pkl')
                       
# get posterior for everyone
c = 0.01 #will be tuned in training
temp_table = temp_table + c*alphas.values
ncol = temp_table.shape[1]
temp_sum = temp_table.iloc[:,:4].sum(axis=1)
for i in range(4):
    ratio = temp_table.iloc[:,i]/temp_sum
    temp_table.iloc[:,i] = ratio
temp_sum = temp_table.iloc[:,4:14].sum(axis=1)
for i in range(4,14):
    ratio = temp_table.iloc[:,i]/temp_sum
    temp_table.iloc[:,i] = ratio
temp_sum = temp_table.iloc[:,14:ncol].sum(axis=1)
for i in range(14,ncol):
    ratio = temp_table.iloc[:,i]/temp_sum
    temp_table.iloc[:,i] = ratio
temp_table.to_pickle('content_posterior.pkl')
del temp_table

############################ browser ##############################
train['browser'] = train['device.device'].apply(lambda x:x['browser'])
browser_counter = Counter(train['browser'])
common_browsers = [key for key in browser_counter.keys() if browser_counter[key]>1000]
np.save('common_browsers.npy',common_browsers)

train['browser'] = train['browser'].apply(lambda x: x if x in common_browsers else 'Other')
temp_table = pd.crosstab(train['fullVisitorId'],train['browser'])
ID_Browser = temp_table.idxmax(axis=1)
ID_Browser = pd.get_dummies(ID_Browser)
ID_Browser.to_pickle('ID_Browser.pkl')

############################ isMobile ##############################
train['mobile'] = train['device.device'].apply(lambda x:x['isMobile']).astype(float)
temp_table = train.groupby(['fullVisitorId']).aggregate({'mobile':['sum','count']})
temp_table.columns = ['s', 'n']

alpha = train['mobile'].sum()+1
beta = train.shape[0]+2-alpha
mobile_parameters = np.array([alpha, beta])
np.save('mobile_parameters.npy', mobile_parameters)
temp_table.to_pickle('mobile_prior.pkl')

c=0.01
temp_table['mobile'] = (c*alpha+temp_table['s'])/(c*alpha+c*beta+temp_table['n'])
temp_table['mobile'].to_pickle('mobile_posterior.pkl')

############################ others  ##############################
train['year'] = train['date'].apply(lambda x: x//10000)
train['month'] = train['date'].apply(lambda x: (x%10000)//100)
train['totals.totalTransactionRevenue'] = train['totals.totalTransactionRevenue'].apply(lambda x: np.exp(x)-1)

def get_time_length(x):
    result = []
    for dicts in x:
        result.append(float(dicts['time']))
    return result
temp = train['hits'].apply(get_time_length)

def average_length(x):
    if len(x)<=1:
        return np.nan
    else:
        return((x[-1]-x[0])/(len(x)-1))
train['average_time'] = temp.apply(average_length)/int(1e5)
train['average_time'] = train['average_time'].fillna(train['average_time'].sum()/(train['average_time'].shape[0]-train['average_time'].apply(np.isnan).sum()))
temp_table = train.groupby(['fullVisitorId'])['average_time'].mean()
temp_table.to_pickle('Train_VisitLength.pkl')

monthly_revenue = train.groupby(['fullVisitorId', 'year', 'month'], as_index=False).aggregate({'totals.totalTransactionRevenue':['sum'],\
                                                                               'geoNetwork.country':['unique']})
monthly_revenue.columns = ['fullVisitorId', 'year', 'month', 'revenue', 'country']
monthly_revenue['country'] = monthly_revenue['country'].apply(lambda x:x[0])
monthly_revenue.to_pickle('Training_Base.pkl')
