def glimpse(start,N=10):
    '''
    Input: 
        start :str, the first several chars of the column names, like 'device', 'geoNetwork' etc.
        N: how many different values to show, default value is 10
    Output:
        result: list. Containing all columns has start.
    This functions shows the columns with less than N unique values. Also returns a whole list with the desired start
    '''
    result=[]
    for names in train.columns.values:
        if names.startswith(start):
            result.append(names)
            counts=train[names].value_counts()
            if len(counts.values)<N:
                print(counts)
                print('\n')
    return result