def custom_length(data_row):
    a1 = len(data_row[0]['customVariables'])
    b1 = len(data_row[0]['customDimensions'])
    c1 = len(data_row[0]['customMetrics'])
    a2 = len(data_row[-1]['customVariables'])
    b2 = len(data_row[-1]['customDimensions'])
    c2 = len(data_row[-1]['customMetrics'])
    return (a1,b1,c1),(a2,b2,c2)