import pandas as pd
import numpy as np

def training_merge(c1=0.00001, c2=0.01):
    '''
    This function merges different sources into a single training set.
    c1 is the weight for prior where fullVisitorId is index
    c2 is the weight for other priors
    '''
    Training = pd.read_pickle('Training_Base.pkl')
    Training['revenue'] = Training['revenue'].apply(lambda x:np.log(x+1))
    
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    Training['month_category'] = Training['month'].apply(lambda x: months[x-1])
    Training = pd.concat([Training, pd.get_dummies(Training['month_category'])], axis=1)
    Training.drop(['month_category'], axis=1, inplace=True)
    
    #join browser information
    ID_Browser = pd.read_pickle('train_ID_Browser.pkl')
    Training.set_index('fullVisitorId', inplace=True)
    Training = Training.join(ID_Browser)
    del ID_Browser
    
    #join content information
    alphas = pd.read_pickle('train_content_alphas.pkl')
    content = pd.read_pickle('train_content_prior.pkl')
    content += alphas*c1
    ncol = content.shape[1]
    temp_sum = content.iloc[:,:4].sum(axis=1)
    for i in range(4):
        ratio = content.iloc[:,i]/temp_sum
        content.iloc[:,i] = ratio
    temp_sum = content.iloc[:,4:14].sum(axis=1)
    for i in range(4,14):
        ratio = content.iloc[:,i]/temp_sum
        content.iloc[:,i] = ratio
    temp_sum = content.iloc[:,14:ncol].sum(axis=1)
    for i in range(14,ncol):
        ratio = content.iloc[:,i]/temp_sum
        content.iloc[:,i] = ratio
    Training = Training.join(content)
    del content, temp_sum
    
    #join mobile information
    mobile = pd.read_pickle('train_mobile_prior.pkl')
    mobile_parameters = np.load('train_mobile_parameters.npy')
    (alpha, beta) = mobile_parameters
    mobile['mobile'] = (c1*alpha+mobile['s'])/(c1*alpha+c1*beta+mobile['n'])
    mobile = mobile['mobile']
    Training = Training.join(mobile)
    del mobile
    
    #join pageView information
    pageView = pd.read_pickle('train_pageView_prior.pkl')
    (alpha, beta) = np.load('train_pageView_parameters.npy')
    pageView['beta'] = c1*beta + pageView['pageview']
    pageView['alpha'] = c1*alpha + pageView['counts']
    pageView['pageview_log'] = pageView['beta']/(pageView['alpha']-1)
    pageView['pageView'] = pageView['pageview_log'].apply(np.exp)
    pageView = pageView.iloc[:,-2:]
    Training = Training.join(pageView)
    del pageView    
              
    #join visitlength information
    visitlength = pd.read_pickle('train_VisitLength.pkl')
    Training = Training.join(visitlength)
    del visitlength
    
    #join country information
    Training.reset_index(inplace=True)
    Training.set_index('country',inplace=True)
    (nu, mu0, alpha, beta) = np.load('train_country_parameters.npy')
    country = pd.read_pickle('train_country_prior.pkl')
    country['country_param1'] = (c2*nu*mu0+country['n']*country['mu'])/(c2*nu+country['n'])
    country['country_param2'] = (c2*alpha+country['raten'])/(c2*alpha+c2*beta+country['raten'])
    country = country.iloc[:,-2:]
    Training = Training.join(country)
    del country
    
    #join monthly time series prediction
    Training.reset_index(inplace=True)
    Training.drop(['index'], axis=1, inplace=True)
    Training.set_index('month_index', inplace=True)
    #month_parameters = pd.read_pickle('train_month_parameters.pkl')
    #Training = Training.join(month_parameters)
    #del month_parameters
    
    Training.reset_index(inplace=True)
    Training.set_index('fullVisitorId', inplace=True)
    Training.drop(['month_index', 'month'], axis=1, inplace=True)
    
    return Training
    
def validate_merge(c1=0.00001, c2=0.01):
    '''
    This function merges different sources into a single training set.
    c1 is the weight for prior where fullVisitorId is index
    c2 is the weight for other priors
    Different from training, the revenue here is already log'ed
    '''
    In_Training = pd.read_pickle('Test_in_Train.pkl')
    
    validation = pd.read_pickle('Testing_Base.pkl')
    validation.set_index('fullVisitorId', inplace=True)
    validation = validation.join(In_Training)
    
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    validation['month_category'] = validation['month'].apply(lambda x: months[x-1])
    validation = pd.concat([validation, pd.get_dummies(validation['month_category'])], axis=1)
    for month in months:
        if month not in validation.columns:
            validation[month]=0
    validation.drop(['month_category'], axis=1, inplace=True)
    
    #join browser information
    ID_Browser = pd.read_pickle('train_ID_Browser.pkl')
    ind = ID_Browser.index[0]
    validation = validation.join(ID_Browser)
    for column in ID_Browser.columns:
        validation.loc[validation['in_training']==False,[column]] = ID_Browser.loc[ind, column]
    del ID_Browser
    
    #join content information
    alphas = pd.read_pickle('train_content_alphas.pkl')
    content = pd.read_pickle('train_content_prior.pkl')
    content += alphas*c1
    ncol = content.shape[1]
    temp_sum = content.iloc[:,:4].sum(axis=1)
    for i in range(4):
        ratio = content.iloc[:,i]/temp_sum
        content.iloc[:,i] = ratio
    temp_sum = content.iloc[:,4:14].sum(axis=1)
    for i in range(4,14):
        ratio = content.iloc[:,i]/temp_sum
        content.iloc[:,i] = ratio
    temp_sum = content.iloc[:,14:ncol].sum(axis=1)
    for i in range(14,ncol):
        ratio = content.iloc[:,i]/temp_sum
        content.iloc[:,i] = ratio
    validation = validation.join(content)
    alphas[:4] = alphas[:4]/np.sum(alphas[:4])
    alphas[4:14] = alphas[4:14]/np.sum(alphas[4:14])
    alphas[14:] = alphas[14:]/np.sum(alphas[14:])
    for column in content.columns:
        validation.loc[validation['in_training']==False,[column]] = alphas[column]
    del content, temp_sum, alphas
    
    #join mobile information
    mobile = pd.read_pickle('train_mobile_prior.pkl')
    mobile_parameters = np.load('train_mobile_parameters.npy')
    (alpha, beta) = mobile_parameters
    mobile['mobile'] = (c1*alpha+mobile['s'])/(c1*alpha+c1*beta+mobile['n'])
    mobile = mobile['mobile']
    validation = validation.join(mobile)
    validation.loc[validation['in_training']==False,['mobile']] = alpha/(alpha + beta)
    del mobile, alpha, beta
    
    #join pageView information
    pageView = pd.read_pickle('train_pageView_prior.pkl')
    (alpha, beta) = np.load('train_pageView_parameters.npy')
    pageView['beta'] = c1*beta + pageView['pageview']
    pageView['alpha'] = c1*alpha + pageView['counts']
    pageView['pageview_log'] = pageView['beta']/(pageView['alpha']-1)
    pageView = pageView.iloc[:,-1:]
    validation = validation.join(pageView)
    validation.loc[validation['in_training']==False,['pageview_log']] = beta/(alpha-1)
    validation['pageView'] = validation['pageview_log'].apply(np.exp)
    del pageView    
              
    #join visitlength information
    visitlength = pd.read_pickle('train_VisitLength.pkl')
    validation = validation.join(visitlength)
    validation.loc[validation['in_training']==False,['average_time']] = validation['average_time'].mean()
    del visitlength
    
    #join country information
    validation.reset_index(inplace=True)
    validation.set_index('country',inplace=True)
    (nu, mu0, alpha, beta) = np.load('train_country_parameters.npy')
    country = pd.read_pickle('train_country_prior.pkl')
    country['country_param1'] = (c2*nu*mu0+country['n']*country['mu'])/(c2*nu+country['n'])
    country['country_param2'] = (c2*alpha+country['raten'])/(c2*alpha+c2*beta+country['raten'])
    country = country.iloc[:,-2:]
    validation = validation.join(country)
    del country
    
    #join monthly time series prediction
    validation.reset_index(inplace=True)
    validation.drop(['index'], axis=1, inplace=True)
    validation.set_index('month_index', inplace=True)
    #month_parameters = pd.read_pickle('train_month_parameters.pkl')
    #validation = validation.join(month_parameters)
    #del month_parameters
    
    validation.set_index('fullVisitorId', inplace=True)
    validation.drop(['month', 'in_training'], axis=1, inplace=True)
    
    return validation
    
    
