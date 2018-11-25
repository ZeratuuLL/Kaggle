def customNull(colname):
    row_num=train.shape[0]
    a=[]
    for i in range(row_num):
        for j in range(len(train[i])):
            if train[i][j][colname]!=[]:
                a.append([i,j])
    return a

