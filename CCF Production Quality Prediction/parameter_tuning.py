import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import log_loss, accuracy_score
from utils import *
import os
import time

NJOBS = 10
N_ESTIMATOR = [100, 250, 500, 1000]
MAX_DEPTH = [6, 8, 10]
LEARNING_RATE = [0.1, 0.05, 0.01]

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

training = pd.read_csv('first_round_training_data.csv')
testing = pd.read_csv('first_round_testing_data.csv')

code = {'Pass':1, 'Good':2, 'Excellent':3, 'Fail':0}
training['new_Quality'] = training['Quality_label'].apply(lambda x : code[x])

feature_list = ["Parameter5", "Parameter7", "Parameter8", "Parameter9", "Parameter10"]
target = 'new_Quality'

training[feature_list] = np.log(training[feature_list].values)/np.log(10)
testing[feature_list] = np.log(testing[feature_list].values)/np.log(10)

for group in range(100):
    name = 'group_%s'%group
    training[name] = 0
    kfold=KFold(n_splits=120, shuffle=True, random_state=group)
    split=kfold.split(training[feature_list])
    i = 0
    for train_index, valid_index in split:
        training.iloc[valid_index,-1] = i
        i+=1

N = 5

#Logistic Regression
skf = StratifiedKFold(n_splits=N)
indices = []
for train_index, test_index in skf.split(training[feature_list], training[[target]]):
    indices.append([train_index, test_index])
t_start = time.time()
results = np.zeros((9, 5))
i = 0
for c in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
    model = LogisticRegression(C=c, multi_class='multinomial', solver='lbfgs')
    train_predict, _ = CV(model, training, testing, indices, feature_list, target, False)
    neg_log_loss = -log_loss(training['new_Quality'], train_predict)
    accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
    offline_score1 = []
    offline_score2 = []
    for j in range(100):
        scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
        offline_score1.append(scores[0])
        offline_score2.append(scores[1])
    results[i, :] = (c, accuracy, neg_log_loss, np.mean(offline_score1), np.mean(offline_score2))
    i += 1
    print('\rLogistic Regression {}/9 finished. Time usage {}'.format(i, time.time()-t_start), end='')
dataframe = pd.DataFrame(data = results, columns = ['C',
                                                    'accuracy',
                                                    'neg_log_loss',
                                                    'offline_score',
                                                    'offline_score_rounded'])
dataframe.to_pickle('./tuning/LR_tune.pkl')
print('\nLogistic Regression Finished')

#random forest
skf = StratifiedKFold(n_splits=N)
indices = []
for train_index, test_index in skf.split(training[feature_list], training[[target]]):
    indices.append([train_index, test_index])
t_start = time.time()
results = np.zeros((12, 6))
i = 0
for n_estimator in N_ESTIMATOR:
    for depth in MAX_DEPTH:
        model = RandomForestClassifier(n_estimators = n_estimator, criterion = 'entropy', max_depth = depth, n_jobs=NJOBS)
        train_predict, _ = CV(model, training, testing, indices, feature_list, target, False)
        neg_log_loss = -log_loss(training['new_Quality'], train_predict)
        accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
        offline_score1 = []
        offline_score2 = []
        for j in range(100):
            scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
            offline_score1.append(scores[0])
            offline_score2.append(scores[1])
        results[i, :] = (n_estimator, depth, accuracy, neg_log_loss, np.mean(offline_score1), np.mean(offline_score2))
        i += 1
    print('\rRandom Forest {}/12 finished. Time usage {}'.format(i, time.time()-t_start), end='')
dataframe = pd.DataFrame(data = results, columns = ['n_estimators',
                                                    'max_depth',
                                                    'accuracy',
                                                    'neg_log_loss',
                                                    'offline_score',
                                                    'offline_score_rounded'])
dataframe.to_pickle('./tuning/RF_tune.pkl')
print('\nRandom Forest Finished')

#XGBoost
skf = StratifiedKFold(n_splits=N)
indices = []
for train_index, test_index in skf.split(training[feature_list], training[[target]]):
    indices.append([train_index, test_index])
t_start = time.time()
results = np.zeros((36, 7))
i = 0
for n_estimators in N_ESTIMATOR:
    for learning_rate in LEARNING_RATE:
        for max_depth in MAX_DEPTH:
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, objective='multi:softmax', verbosity=0, verbose_eval=False, n_jobs=NJOBS)
            train_predict, _ = CV(model, training, testing, indices, feature_list, target, False)
            neg_log_loss = -log_loss(training['new_Quality'], train_predict)
            accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
            offline_score1 = []
            offline_score2 = []
            for j in range(100):
                scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
                offline_score1.append(scores[0])
                offline_score2.append(scores[1])
            results[i, :] = (n_estimators, learning_rate, max_depth, accuracy, neg_log_loss, np.mean(offline_score1), np.mean(offline_score2))
            i += 1
            print('\rXGBoost {}/36 finished. Time usage {}'.format(i, time.time()-t_start), end='')
dataframe = pd.DataFrame(data = results, columns = ['n_estimators',
                                                    'learning_rate',
                                                    'max_depth',
                                                    'accuracy',
                                                    'neg_log_loss',
                                                    'offline_score',
                                                    'offline_score_rounded'])
dataframe.to_pickle('./tuning/XGB_tune.pkl')
print('\nXGBoost Finished')

#Catboost
skf = StratifiedKFold(n_splits=N)
indices = []
for train_index, test_index in skf.split(training[feature_list], training[[target]]):
    indices.append([train_index, test_index])
t_start = time.time()
results = np.zeros((36, 7))
i = 0
for n_estimators in N_ESTIMATOR:
    for learning_rate in LEARNING_RATE:
        for max_depth in MAX_DEPTH:
            model = CatBoostClassifier(iterations=n_estimators, depth=max_depth, l2_leaf_reg=10, learning_rate=learning_rate, silent = True, loss_function = 'MultiClass', task_type = 'GPU')
            train_predict, _ = CV(model, training, testing, indices, feature_list, target, False)
            neg_log_loss = -log_loss(training['new_Quality'], train_predict)
            accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
            offline_score1 = []
            offline_score2 = []
            for j in range(100):
                scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
                offline_score1.append(scores[0])
                offline_score2.append(scores[1])
            results[i, :] = (n_estimators, learning_rate, max_depth, accuracy, neg_log_loss, np.mean(offline_score1), np.mean(offline_score2))
            i += 1
            print('\rCatBoost {}/36 finished. Time usage {}'.format(i, time.time()-t_start), end='')
dataframe = pd.DataFrame(data = results, columns = ['iterations',
                                                    'learning_rate',
                                                    'max_depth',
                                                    'accuracy',
                                                    'neg_log_loss',
                                                    'offline_score',
                                                    'offline_score_rounded'])
dataframe.to_pickle('./tuning/catboost_tune.pkl')
print('\nCatBoost Finished')

#Lightgbm
skf = StratifiedKFold(n_splits=N)
indices = []
for train_index, test_index in skf.split(training[feature_list], training[[target]]):
    indices.append([train_index, test_index])
t_start = time.time()
results = np.zeros((36, 7))
i = 0
for n_estimators in N_ESTIMATOR:
    for learning_rate in LEARNING_RATE:
        for max_depth in MAX_DEPTH:
            model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, objective='multiclass', n_jobs=NJOBS)
            train_predict, _ = CV(model, training, testing, indices, feature_list, target, False)
            neg_log_loss = -log_loss(training['new_Quality'], train_predict)
            accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
            offline_score1 = []
            offline_score2 = []
            for j in range(100):
                scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
                offline_score1.append(scores[0])
                offline_score2.append(scores[1])
            results[i, :] = (n_estimators, learning_rate, max_depth, accuracy, neg_log_loss, np.mean(offline_score1), np.mean(offline_score2))
            i += 1
            print('\rLightgbm {}/36 finished. Time usage {}'.format(i, time.time()-t_start), end='')
dataframe = pd.DataFrame(data = results, columns = ['n_estimators',
                                                    'learning_rate',
                                                    'max_depth',
                                                    'accuracy',
                                                    'neg_log_loss',
                                                    'offline_score',
                                                    'offline_score_rounded'])
dataframe.to_pickle('./tuning/lightgbm_tune.pkl')
print('\nLightgbm Finished')