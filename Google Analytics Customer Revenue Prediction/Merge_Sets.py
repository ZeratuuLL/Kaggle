def Merge(c1=0.00001, c2=0.01):
    '''
    This function merges different sources into a single training set.
    c1 is the weight for prior where fullVisitorId is index
    c2 is the weight for other priors
    '''
    Training = pd.read_pickle('Training_Base.pkl')
    Training['revenue'] = Training['revenue'].apply(lambda x:np.log(x+1))
    
    #join browser information
    ID_Browser = pd.read_pickle('ID_Browser.pkl')
    Training.set_index('fullVisitorId', inplace=True)
    Training = Training.join(ID_Browser)
    del ID_Browser
    
    #join content information
    alphas = pd.read_pickle('content_alphas.pkl')
    content = pd.read_pickle('content_prior.pkl')
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
    mobile = pd.read_pickle('mobile_prior.pkl')
    mobile_parameters = np.load('mobile_parameters.npy')
    (alpha, beta) = mobile_parameters
    mobile['mobile'] = (c1*alpha+mobile['s'])/(c1*alpha+c1*beta+mobile['n'])
    mobile = mobile['mobile']
    Training = Training.join(mobile)
    del mobile
    
    #join pageView information
    pageView = pd.read_pickle('pageView_prior.pkl')
    (alpha, beta) = np.load('pageView_parameters.npy')
    pageView['beta'] = c1*beta + pageView['pageview']
    pageView['alpha'] = c1*alpha + pageView['counts']
    pageView['pageview_log'] = pageView['beta']/(pageView['alpha']-1)
    pageView['pageView'] = pageView['pageview_log'].apply(np.exp)
    pageView = pageView.iloc[:,-2:]
    Training = Training.join(pageView)
    del pageView    
    
    #join country information
    Training.reset_index(inplace=True)
    Training.set_index('country',inplace=True)
    (nu, mu0, alpha, beta) = np.load('country_parameters.npy')
    country = pd.read_pickle('country_prior.pkl')
    country['country_param1'] = (c2*nu*mu0+country['n']*country['mu'])/(c2*nu+country['n'])
    country['country_param2'] = (c2*alpha+country['raten'])/(c2*alpha+c2*beta+country['raten'])
    country = country.iloc[:,-2:]
    Training = Training.join(country)
    
    Training.set_index('fullVisitorId', inplace=True)
    Training.drop(['country'], inplace=True)
    
    return Training
    
    
    
    
