def find_ind(some_list):
    '''
    Input: 
        some_list: list-like, containing some IDs, cannot be a single ID
    Output: 
        the indices of IDs, which have repetations in training set, in some_list, but not grouped(or sorted) by ID
    '''
    some_list = set(some_list)
    ins = train['fullVisitorId'].apply(lambda x: x in some_list)
    ind=list(ins.index[ins.values])
    return ind
