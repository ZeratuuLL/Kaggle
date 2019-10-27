import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, accuracy_score
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool

import matplotlib.pyplot as plt

pca = PCA()

text_file=open('training.txt','w')

MAX_DEPTH = [6, 8, 10]
LEARNING_RATE = [0.01, 0.005]
ITERATIONS = [1500, 1750, 2000, 2250]
L2_NORM = [0, 15, 30]
L1 = len(MAX_DEPTH)
L2 = len(LEARNING_RATE)
L3 = len(ITERATIONS)
L4 = len(L2_NORM)

# read in data
training = pd.read_csv('first_round_training_data.csv')
testing = pd.read_csv('first_round_testing_data.csv')
features = ["Parameter5","Parameter6","Parameter7","Parameter8","Parameter9","Parameter10"]

training[features] = np.log(training[features].values)/np.log(10)
testing[features] = np.log(testing[features].values)/np.log(10)

code = {'Pass':1, 'Good':2, 'Excellent':3, 'Fail':0}
training['new_Quality'] = training['Quality_label'].apply(lambda x : code[x])

N = 10

skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=302)
indices = []
for train_index, test_index in skf.split(training[features], training[['new_Quality']]):
    indices.append([train_index, test_index])

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
'''
#No encoding No PCA
text_file.write('No encoding no PCA\n')
new_features = ['Parameter'+str(i) for i in [5, 7, 8, 9, 10]]
train_all, test_all = training.copy(), testing.copy()
new_features = new_features

#No encoding at all
text_file.write('No Encoding\n')
new_features = ['Parameter'+str(i) for i in range(5, 11)]
train_all, test_all = training.copy(), testing.copy()
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:5]
text_file.write('Number of features : {}\n'.format(len(new_features)))
'''
#target mean encoding
text_file.write('All target mean encoding\n')
train_all, test_all, new_features = get_all_encoding(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:15]
text_file.write('Number of features : {}\n'.format(len(new_features)))
'''
#WoE encoding
text_file.write('All WoE encoding\n')
train_all, test_all, new_features = get_all_WoE(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:14]
text_file.write('Number of features : {}\n'.format(len(new_features)))
'''
def i_to_tuple(i):
    n1 = math.floor(i/(L2*L3*L4))
    n2 = math.floor(i/(L3*L4)) % L2
    n3 = math.floor(i/L4) % L3
    n4 = i % L4
    return (n1, n2, n3, n4)    

def my_CV(params):
    i, iterations, depth, learning_rate, l2_leaf_reg = params
    train_index = indices[i][0]
    test_index = indices[i][1]
    model = CatBoostClassifier(iterations=iterations, 
                               depth=depth, 
                               learning_rate=learning_rate, 
                               silent=True, 
                               loss_function='MultiClass', 
                               l2_leaf_reg=l2_leaf_reg)
    X_train = train_all.loc[train_index, new_features]
    y_train = train_all.loc[train_index, ['new_Quality']]
    X_test = train_all.loc[test_index, new_features]
    y_test = train_all.loc[test_index, ['new_Quality']]
    model.fit(X=X_train, y=y_train, eval_set=(X_test, y_test))
    train_probs = model.predict_proba(X_train)
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_neg_log_loss = -log_loss(y_train, train_probs)
    test_probs = model.predict_proba(X_test)
    test_neg_log_loss = -log_loss(y_test, test_probs)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return (train_accuracy, train_neg_log_loss, test_accuracy, test_neg_log_loss)

pool = Pool(processes = N)
results = np.zeros((L1*L2*L3*L4, 4))

for i in tqdm(range(L1*L2*L3*L4)):
    n1, n2, n3, n4 = i_to_tuple(i)
    iterations = ITERATIONS[n3]
    depth = MAX_DEPTH[n1]
    learning_rate = LEARNING_RATE[n2]
    l2_leaf_reg = L2_NORM[n4]
    results[i, :] = np.mean(np.array(pool.map(my_CV, [(j, iterations, depth, learning_rate, l2_leaf_reg) for j in range(N)])), axis=0)

index = np.argsort(-results[:, 3])#sort by test neg log loss, decreasing order
n1, n2, n3, n4 = i_to_tuple(index[0])
text_file.write('Best configuration is : {} iterations,\n\t{} depth,\n\t{} learning rate and\n\t{} l2 regularization\n'.format(ITERATIONS[n3], MAX_DEPTH[n1], LEARNING_RATE[n2], L2_NORM[n4]))

text_file.write('Top five train accuracy : {}\n'.format(results[index[:5], 0]))
text_file.write('Top five train neg log loss : {}\n'.format(results[index[:5], 1]))
text_file.write('Top five test accuracy : {}\n'.format(results[index[:5], 2]))
text_file.write('Top five test neg log loss: {}\n'.format(results[index[:5], 3]))

text_file.close()
np.save('results.npy', results)
