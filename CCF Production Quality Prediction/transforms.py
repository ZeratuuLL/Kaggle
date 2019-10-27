import numpy as np
import pandas as pd
from collections import defaultdict

def to_category_fit(values):
    '''
    This function takes a group of values as input and returns a dictionary, whose keys are original values
    '''
    codebook = {}
    count = 0
    for i in values:
        if i not in codebook.keys():
            count += 1
            codebook[i] = count
    return codebook

def to_category_apply(values, codebook):
    '''
    This function takes a group of values and the codebook to use as inputs, returns the encoded variable
    This codebook should be the output of to_category_fit()
    '''
    codes = []
    all_keys = list(codebook.keys())
    for value in values:
        if value in codebook.keys():
            codes.append(codebook[value])
        else:
            index = np.argmin((np.array(all_keys) - value)**2)
            codes.append(codebook[all_keys[index]])
    return codes

def all_to_category(training, validation, col_list):
    new_features = []
    for column in col_list:
        codebook = to_category_fit(training[column].values)
        new_name = '{}_Categorical'.format(column)
        new_features.append(new_name)
        training[new_name] = to_category_apply(training[column].values, codebook)
        validation[new_name] = to_category_apply(validation[column].values, codebook)
    return training, validation, new_features

def target_mean_encoding_fit(variables, targets):
    '''
    This function returns the codebook to target mean encoding of the input categorical variable
    Input variable can be integers, strings or values, targets are the labels, which should be float numbers
    
    It returns the codebook from variable values to its encoding
    '''
    df = pd.DataFrame({'variable': variables, 'targets':targets})
    codebook = defaultdict(lambda : np.mean(targets))
    df = df.groupby(['variable'], as_index=False).agg('mean')
    for i in range(df.shape[0]):
        codebook[df.iloc[i, 0]] = df.iloc[i, 1]
    return codebook

def target_mean_encoding_apply(variables, codebook):
    '''
    This function returns the target mean encoding of the input categorical variable
    Input variable can be integers, strings or values, codebook is the rules to follow to get encodings.
    This codebook should be the output of traget_mean_encoding_fit()
    
    It returns the encoded values
    '''
    result = []
    all_keys = list(codebook.keys())
    for value in variables:
        if value in codebook.keys():
            result.append(codebook[value])
        else:
            index = np.argmin((np.array(all_keys) - value)**2)
            result.append(codebook[all_keys[index]])
    return result

def weighted_average_encoding_fit(variables, targets):
    '''
    This function returns some basic elements for calculating weighted average encoding of categorical variable
    Input variable can be integers, strings or values and target is the target label
    Output are the codebook of target mean encoding and the counts for each variable
    '''
    codebook = target_mean_encoding_fit(variables, targets)
    counts = {}
    for value in set(variables):
        counts.update({value : np.sum(variables==value)})
    return codebook, counts

def weighted_average_encoding_apply(variable, codebook, counts, k, f):
    '''
    This function returns the weighted average encoding of the input categorical variable
    Input variable can be integers, strings or values, codebook is the rules to follow to get encodings.
    Input codebook and counts should be the output of weighted_average_mean_encoding_fit()
    Input counts should be a dictionary whose values are the count of the keys in training set.
    
    It returns the encoded values
    '''
    overall_mean = codebook['this key cannot exist']
    target_means = []
    real_counts = []
    all_keys = list(codebook.keys())
    for value in variable:
        if value in codebook.keys():
            target_means.append(codebook[value])
            real_counts.append(counts[value])
        else:
            index = np.argmin((np.array(all_keys) - value)**2)
            target_means.append(codebook[all_keys[index]])
            real_counts.append(counts[all_keys[index]])
    weights = 1/(1+np.exp((k-np.array(real_counts))/f))
    result = weights * target_means + (1-weights) * overall_mean
    return result

def get_target_mean_encoding(training, validation, col_list, target_name):
    '''
    Use previous funcionts to transform the columns in col_list to their mean target encoding
    The transformation is in both training and validation set
    target_name is the name of column which is the label
    
    Return the transformed training/validation datasets
    '''
    for column in col_list:
        codebook = target_mean_encoding_fit(training[column].values, training[target_name].values)
        training.loc[:, column] = target_mean_encoding_apply(training[column], codebook)
        validation.loc[:, column] = target_mean_encoding_apply(validation[column], codebook)
    
    return training, validation

def get_weighted_average_encoding(training, validation, col_list, target_name, k, f):
    '''
    Use previous funcionts to transform the columns in col_list to their mean target encoding
    The transformation is in both training and validation set
    target_name is the name of column which is the label
    k is inflection point and f is steepness
    
    Return the transformed training/validation datasets
    '''
    for column in col_list:
        codebook, counts = weighted_average_encoding_fit(training[column].values, training[target_name].values)
        training.loc[:, column] = weighted_average_encoding_apply(training[column], codebook, counts, k, f)
        validation.loc[:, column] = weighted_average_encoding_apply(validation[column], codebook, counts, k, f)
    
    return training, validation

def get_all_encoding(training, validation, col_list):
    new_features = []
    for column in col_list:
        for level in ['Fail', 'Pass', 'Good', 'Excellent']:
            codebook = target_mean_encoding_fit(training[column].values, training['Quality_label'].values==level)
            new_name = '{}_{}'.format(column, level)
            new_features.append(new_name)
            training[new_name] = target_mean_encoding_apply(training[column], codebook)
            validation[new_name] = target_mean_encoding_apply(validation[column], codebook)
    return training, validation, new_features

def weight_of_evidence_fit(variables, targets, base=0.5):
    '''
    This function returns the weight of evidence (WoE) encoding for the input categorical variable
    Input variable can be integers, strings or values
    Input targets must be 0-1 one hot encoding
    
    It returns the codebook from values to WoE
    '''
    total_1 = np.sum(targets==1)
    total_0 = np.sum(targets==0)
    codebook = defaultdict(lambda : np.mean(targets))
    df = pd.DataFrame({'variable': variables, 'targets':targets})
    df = pd.crosstab(variables, targets)
    values = df.index.values
    for i in range(df.shape[0]):
        in_group_1 = df.iloc[i, 1]
        in_group_0 = df.iloc[i, 0]
        codebook.update({values[i] : np.log((in_group_0+base)/(in_group_1+base)) + np.log(total_1/total_0)})
    return codebook

def weight_of_evidence_apply(variables, codebook):
    result = []
    all_keys = list(codebook.keys())
    for value in variables:
        if value in codebook.keys():
            result.append(codebook[value])
        else:
            index = np.argmin((np.array(all_keys) - value)**2)
            result.append(codebook[all_keys[index]])
    return np.array(result)

def get_all_WoE(training, validation, col_list, base=0.5):
    new_features = []
    for column in col_list:
        for level in ['Fail', 'Pass', 'Good', 'Excellent']:
            codebook = weight_of_evidence_fit(training[column], training['Quality_label']==level, base)
            new_name = '{}_WoE_{}'.format(column, level)
            new_features.append(new_name)
            training[new_name] = weight_of_evidence_apply(training[column].values, codebook)
            validation[new_name] = weight_of_evidence_apply(validation[column].values, codebook)
    return training, validation, new_features