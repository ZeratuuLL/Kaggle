import pandas as pd

train = pd.read_pickle('Train_Saved.pkl')

def custom_length(x):
    keys = ['customVariables', 'customDimensions', 'customMetrics']
    result = []
    for dicts in x:
        temp = ()
        for j in keys:
            temp += (len(dicts[j]),)
        result.append(temp)
    return result

############################# totals #############################
train['totals.totalTransactionRevenue'] = train['totals.totalTransactionRevenue'].apply(lambda x: np.float(x))
train['totals.haveRevenue'] = train['totals.totalTransactionRevenue'].apply(lambda x: not np.isnan(x))
pd.crosstab(train['eCommerceAction.Purchase']>0, train['totals.haveRevenue'])
