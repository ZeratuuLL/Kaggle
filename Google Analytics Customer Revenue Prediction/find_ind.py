def find_ind(some_list):
    '''
    Input: 
        some_list: list-like, containing some IDs
    Output: 
        the indices of IDs, which have repetations in training set, in some_list
    '''
    ind=set()
    some_list=set(some_list)
    for ID in some_list:
        indices=train[train['fullVisitorId']==ID].index.values
        if len(indices)>=2:
            ind=ind | set(indices)
    return list(ind)