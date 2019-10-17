import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, accuracy_score
from collections import defaultdict
from tqdm import tqdm
from transforms import *
import seaborn as sns
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool

import matplotlib.pyplot as plt

pca = PCA()

MAX_DEPTH = [2, 3, 4, 6, 8, 10]
LEARNING_RATE = [0.1, 0.005, 0.001, 0.0005]
MIN_SAMPLE_IN_LEAF = [4, 8, 12, 16, 20]
L2_NORM = [0, 5, 10]
L1 = len(MAX_DEPTH)
L2 = len(LEARNING_RATE)
L3 = len(MIN_SAMPLE_IN_LEAF)
L4 = len(L2_NORM)

# read in data
training = pd.read_csv('first_round_training_data.csv')
testing = pd.read_csv('first_round_testing_data.csv')
features = ["Parameter5","Parameter6","Parameter7","Parameter8","Parameter9","Parameter10"]

training[features] = np.log(training[features].values)/np.log(10)
testing[features] = np.log(testing[features].values)/np.log(10)

code = {'Pass':1, 'Good':2, 'Excellent':3, 'Fail':0}
training['new_Quality'] = training['Quality_label'].apply(lambda x : code[x])

N = 5

skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=302)
indices = []
for train_index, test_index in skf.split(training[features], training[['new_Quality']]):
    indices.append([train_index, test_index])
    
for group in range(100):
    name = 'group_%s'%group
    training[name] = 0
    kfold=KFold(n_splits=120, shuffle=True, random_state=group)
    split=kfold.split(training[features])
    i = 0
    for train_index, valid_index in split:
        training.iloc[valid_index,-1] = i
        i+=1
        
def normalize(values):
    values = values/0.02
    upper = np.ceil(values)
    lower = np.floor(values)
    for i in range(values.shape[0]):
        smallest_error = 10000
        for a in [upper[i, 0], lower[i, 0]]:
            for b in [upper[i, 1], lower[i, 1]]:
                for c in [upper[i, 2], lower[i, 2]]:
                    for d in [upper[i, 3], lower[i, 3]]:
                        if a+b+c+d == 50:
                            new_value = np.array([a, b, c, d])
                            new_error = np.mean(np.abs(new_value - values[i, :]))
                            if new_error < smallest_error:
                                smallest_error = new_error
                                best_option = new_value
        values[i, :] = best_option.copy()
    return values*0.02

def approx_score(probs, labels, groups):
    labels = labels.reshape(-1, 1)
    groups = groups.reshape(-1, 1)
    matrix = np.concatenate([probs, labels==0, labels==1, labels==2, labels==3, groups], axis=1)
    temp_frame = pd.DataFrame(data=matrix, 
                              columns=['prob1', 'prob2', 'prob3', 'prob4', 'label1', 'label2', 'label3', 'label4', 'group']
                             )
    temp_frame = temp_frame.groupby(['group']).mean()
    matrix = temp_frame.values
    mae1 = np.mean(np.abs(matrix[:, range(4)] - matrix[:, range(4, 8)]))
    score1 = 1/(1+10*mae1)
    rounded = normalize(matrix[:, range(4)])
    mae2 = np.mean(np.abs(rounded - matrix[:, range(4, 8)]))
    score2 = 1/(1+10*mae2)
    return score1, score2

def i_to_tuple(i):
    n1 = math.floor(i/(L2*L3*L4))
    n2 = math.floor(i/(L3*L4)) % L2
    n3 = math.floor(i/L4) % L3
    n4 = i % L4
    return (n1, n2, n3, n4)    

def my_CV(i):
    n1, n2, n3, n4 = i_to_tuple(i)
    min_sample_in_leaf = MIN_SAMPLE_IN_LEAF[n3]
    depth = MAX_DEPTH[n1]
    learning_rate = LEARNING_RATE[n2]
    l2_leaf_reg = L2_NORM[n4]
    model = CatBoostClassifier(iterations = 3000, 
                               depth = depth, 
                               learning_rate = learning_rate, 
                               silent = True, 
                               loss_function = 'MultiClass', 
                               l2_leaf_reg = l2_leaf_reg,
                               task_type = 'GPU',
                               od_type = 'Iter',
                               od_wait = 50, 
                               min_data_in_leaf = min_sample_in_leaf)
    probs = np.zeros((6000, 4))
    predictions = np.zeros(6000)
    train_neg_log_loss = []
    train_accuracy = []
    best_iterations = []
    for j in range(N):
        train_index = indices[j][0]
        test_index = indices[j][1]
        X_train = train_all.loc[train_index, new_features]
        y_train = train_all.loc[train_index, ['new_Quality']]
        X_test = train_all.loc[test_index, new_features]
        y_test = train_all.loc[test_index, ['new_Quality']]
        model.fit(X=X_train, y=y_train, eval_set=(X_test, y_test))
        probs[test_index, :] = model.predict_proba(X_test)
        predictions[test_index] = model.predict(X_test).reshape(-1)
        train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))
        train_neg_log_loss.append(-log_loss(y_train, model.predict_proba(X_train)))
        best_iterations.append(model.get_best_iteration())
    test_neg_log_loss = -log_loss(training['new_Quality'], probs)
    test_accuracy = accuracy_score(training['new_Quality'], predictions)
    offline_score1 = []
    offline_score2 = []
    for j in range(100):
        scores = approx_score(probs, training['new_Quality'].values, training['group_{}'.format(j)].values)
        offline_score1.append(scores[0])
        offline_score2.append(scores[1])
    return (np.mean(train_neg_log_loss), 
            np.std(train_neg_log_loss),
            np.mean(train_accuracy), 
            np.std(train_accuracy),
            test_neg_log_loss, 
            test_accuracy, 
            np.mean(offline_score1),
            np.mean(offline_score2),
            np.std(offline_score1),
            np.max(offline_score1) - np.min(offline_score1),
            min_sample_in_leaf,
            depth,
            learning_rate,
            l2_leaf_reg, 
            np.mean(best_iterations),
            np.std(best_iterations))

print('No Encoding No PCA, features 5, 7, 8, 9, 10')
new_features = ['Parameter'+str(i) for i in [5, 7, 8, 9, 10]]
train_all, test_all = training.copy(), testing.copy()
new_features = new_features
filename = 'Original_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'min_sample_in_leaf',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Only PCA, 5 components')
new_features = ['Parameter'+str(i) for i in range(5, 11)]
train_all, test_all = training.copy(), testing.copy()
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:5]
filename = 'PCA_Only_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Only PCA, 6 components')
new_features = ['Parameter'+str(i) for i in range(5, 11)]
train_all, test_all = training.copy(), testing.copy()
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:6]
filename = 'PCA_Only_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Target Mean Encoding, PCA, 9 components,0.913')
train_all, test_all, new_features = get_all_encoding(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:9]
filename = 'Target_Mean_Encoding_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Target Mean Encoding, PCA, 11 components, 0.950')
train_all, test_all, new_features = get_all_encoding(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:11]
filename = 'Target_Mean_Encoding_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Target Mean Encoding, PCA, 15 components, 0.989')
train_all, test_all, new_features = get_all_encoding(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:15]
filename = 'Target_Mean_Encoding_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Weight Of Evidence Encoding, PCA, 10 components, 0.910')
train_all, test_all, new_features = get_all_WoE(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:10]
filename = 'Weight_of_Evidence_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Weight Of Evidence Encoding, PCA, 13 components, 0.958')
train_all, test_all, new_features = get_all_WoE(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:13]
filename = 'Weight_of_Evidence_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')

print('Weight Of Evidence Encoding, PCA, 17 components, 0.989')
train_all, test_all, new_features = get_all_WoE(training, testing, features)
new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
train_all[new_features] = new_values[:6000, :].copy()
test_all[new_features] = new_values[6000:, :].copy()
new_features = new_features[:17]
filename = 'Weight_of_Evidence_{}'.format(len(new_features))
results = np.zeros((L1*L2*L3*L4, 16))
for i in tqdm(range(L1*L2*L3*L4)):
    results[i, :] = my_CV(i)
results = pd.DataFrame(data=results, columns=['train_neg_log_loss_mean', 
                                              'train_neg_log_loss_std', 
                                              'train_accuracy_mean', 
                                              'train_accuracy_std', 
                                              'validate_neg_log_loss', 
                                              'validate_accuracy', 
                                              'validate_offline_score_mean',
                                              'validate_offline_score_rounded',
                                              'validate_offline_score_std',
                                              'validate_offline_score_range',
                                              'Iterations',
                                              'depth',
                                              'learning_rate',
                                              'l2_norm',
                                              'best_iterations_mean',
                                              'best_iterations_std'])
results.to_pickle(filename+'.pkl')


