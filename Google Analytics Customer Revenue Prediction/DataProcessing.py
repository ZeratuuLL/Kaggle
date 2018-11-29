import pandas as pd
from collections import Counter
import numpy as np

train = pd.read_pickle('Train_Saved.pkl')

############################# totals #############################
train['totals.totalTransactionRevenue'] = train['totals.totalTransactionRevenue'].astype(float).fillna(0).apply(lambda x: np.log(x+1))

train['totals.haveRevenue'] = train['totals.totalTransactionRevenue']>0

# relationship between number of pages and rate of purchase
train['totals.pageviews'] = train['totals.pageviews'].astype(float)

# relationship between geo information and shopping rate, this is going to be modeled by binomial
country_rate = train.groupby(['geoNetwork.country'])['totals.haveRevenue'].mean()
continent_rate = train.groupby(['geoNetwork.continent'])['totals.haveRevenue'].mean()
subcontinent_rate = train.groupby(['geoNetwork.subContinent'])['totals.haveRevenue'].mean()
###########################################################################
###########################################################################
###                                                                     ###
###                                                                     ###
###                  Here is the code for calculating                   ###
###            posterior MLE of buying rate at different levels         ###
###                            Shitong                                  ###
###                                                                     ###                     
###########################################################################
###########################################################################

# relationship between geo information and shopping revenue, this is going to be modeled by (what prior)?
country_value = train.groupby(['geoNetwork.country'])['totals.totalTransactionRevenue'].mean()
continent_value = train.groupby(['geoNetwork.continent'])['totals.totalTransactionRevenue'].mean()
subcontinent_value = train.groupby(['geoNetwork.subContinent'])['totals.totalTransactionRevenue'].mean()
###########################################################################
###########################################################################
###                                                                     ###
###                                                                     ###
###                  Here is the code for calculating                   ###
###            posterior MLE of average revenue per puchase             ###
###                         at different levels                         ###
###                            Shitong                                  ###
###                                                                     ###                     
###########################################################################
###########################################################################

def post_est(mu0,nu,x,n,alpha):
    return (alpha*nu*mu0+(1-alpha)*n*x)/(alpha*nu+(1-alpha)*n)

def post_est_dic(x,alpha):
    nu = train.shape[0]
    temp_mu = train['geoNetwork.'+x].mean()
    temp_post_est = dict()
    for i in train['geoNetwork.'+x].unique():
        temp = train[train['geoNetwork.'+x]==i]['totals.totalTransactionRevenue']
        x_bar = temp.mean()
        n_temp = temp.count()
        post_temp = post_est(temp_mu,nu,x_bar,n_temp,alpha)
        temp_post_est.update({i:post_temp})
     return temp_post_est



# page visit number
###########################################################################
###########################################################################
###                                                                     ###
###                                                                     ###
###                Here is the code for calculating                     ###
###          posterior MLE of page visit number and its parameter       ###
###                            Tongyi                                   ###
###                                                                     ###                     
###########################################################################
###########################################################################


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

# calculate prior
temp_table = pd.DataFrame({'fullVisitorId':train['fullVisitorId'].values.tolist()})
for i in range(len(unique_content1)):
    temp_table['content1.'+unique_content1[i]] = content1_count.apply(lambda x: x[i])
for i in range(len(unique_content2)):
    temp_table['content2.'+unique_content2[i]] = content2_count.apply(lambda x: x[i])
for i in range(len(unique_content3)):
    temp_table['content3.'+unique_content3[i]] = content3_count.apply(lambda x: x[i])

# get total content counts for each user
temp_table = temp_table.groupby(['fullVisitorId']).sum()#After this fullVisitorId becomes Index
alphas = temp_table.sum()+1 #This is the prior parameters for multinomial 

# get posterior for everyone
c = 0.01 #will be tuned in training
temp_table = temp_table + c*alphas.values
temp_table.iloc[:,:4] = temp_table.iloc[:,:4]/temp_table.iloc[:,:4].sum(axis=1)
temp_table.iloc[:,4:14] = temp_table.iloc[:,4:14]/temp_table.iloc[:,4:14].sum(axis=1)
temp_table.iloc[:,14:] = temp_table.iloc[:,14:]/temp_table.iloc[:,14:].sum(axis=1)

############################ geoNetwork columns  ##############################


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
