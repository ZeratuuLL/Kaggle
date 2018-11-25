import pandas as pd

train = pd.read_pickle('Train_Saved.pkl')

def custom_length(x):
    keys = ['customVariables', 'customDimensions', 'customMetrics']
    result = ()
    for i in [0,-1]:
        temp = ()
        for j in keys:
            temp += (len(x[i][j]),)
        result += temp
    return result

############################# totals #############################
train['totals.totalTransactionRevenue'].astype(float, inplace=True)
