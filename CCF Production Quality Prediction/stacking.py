from utils import *
from transforms import *
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

N = 5

training = pd.read_csv('first_round_training_data.csv')
testing = pd.read_csv('first_round_testing_data.csv')
features = ["Parameter5","Parameter7","Parameter8","Parameter9","Parameter10"]
target = 'new_Quality'

L = np.zeros((training.shape[0], 4))
labels = ['Fail', 'Pass', 'Good', 'Excellent']
for i in range(4):
    L[:, i] = (training['Quality_label'].values==labels[i]).astype('float')

training[features] = np.log(training[features].values)/np.log(10)
testing[features] = np.log(testing[features].values)/np.log(10)

code = {'Pass':1, 'Good':2, 'Excellent':3, 'Fail':0}
training['new_Quality'] = training['Quality_label'].apply(lambda x : code[x])

type_to_num = {'Original':0, 'PCA':1, 'TargetMean':2, 'WoE':3}
num_to_type = dict(zip(type_to_num.values(), type_to_num.keys()))
    
for group in range(100):
    name = 'group_%s'%group
    training[name] = 0
    kfold=KFold(n_splits=120, shuffle=True, random_state=group)
    split=kfold.split(training[features])
    i = 0
    for train_index, valid_index in split:
        training.iloc[valid_index,-1] = i
        i+=1

# this is the list of candidates
candidates_list = [#(LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs'), 'LR'),
                   (RandomForestClassifier(n_estimators = 250, criterion = 'entropy', max_depth = 6, n_jobs=10), 'RF'),
                   (XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=6, objective='multi:softmax', verbosity=0, verbose_eval=False, n_jobs=10), 'XGB'),
                   (CatBoostClassifier(iterations=1000, depth=6, l2_leaf_reg=10, learning_rate=0.01, silent = True, loss_function = 'MultiClass', task_type = 'GPU'), 'Cat'),
                   #(LGBMClassifier(n_estimators=250, learning_rate=0.01, max_depth=6, objective='multiclass', n_jobs=10), 'Light')
                  ]
name_list = []

if len(name_list) == 0 :
    def train(candidate):
        '''
        params are parameters to train the model. It should be a tuple of (train_type, num_features, depth, lr, l2)
        '''
        skf = StratifiedKFold(n_splits=N)
        indices = []
        for train_index, test_index in skf.split(training[features], training[[target]]):
            indices.append([train_index, test_index])
        model, name = candidate
        train_predict, test_predict = CV(model, training, testing, indices, features, 'new_Quality', True)
        neg_log_loss = -log_loss(training['new_Quality'], train_predict)
        accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
        name_list.append(name)
        np.save('./stacking/'+name_list[-1]+'_train.npy', train_predict[:, :3])
        np.save('./stacking/'+name_list[-1]+'_test.npy', test_predict[:, :3])
        offline_score1 = []
        offline_score2 = []
        for j in range(100):
            scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
            offline_score1.append(scores[0])
            offline_score2.append(scores[1])
        return np.mean(offline_score1), np.mean(offline_score2), neg_log_loss, accuracy, 
    
    train_results = []
    for candidate in tqdm.tqdm(candidates_list):
        train_results.append(train(candidate))
    train_results = np.array(train_results)
    train_results = pd.DataFrame(data = train_results, columns = ['validate_offline_score_mean',
                                                                  'validate_offline_score_rounded',
                                                                  'neg_log_loss',
                                                                  'accuracy'])
    train_results['train_type'] = [candidate[1] for candidate in candidates_list]
    train_results.to_pickle('./stacking/stacking_train.pkl')

print('name list is {}'.format(name_list))
train_list = [np.load('./stacking/{}_train.npy'.format(name)) for name in name_list] + [training[features].values]
test_list = [np.load('./stacking/{}_test.npy'.format(name)) for name in name_list] + [testing[features].values]
column_list = ['column_{}'.format(i) for i in range(3*len(name_list))] + features

new_train_data = pd.DataFrame(data=np.concatenate(train_list, 1), columns=column_list)
new_test_data = pd.DataFrame(data=np.concatenate(test_list, 1), columns=column_list)

# now we have 'new features', we will use CV again to select hyper-paraemters
# First we will do the split again
skf = StratifiedKFold(n_splits=N, shuffle=True)
indices = []
for train_index, test_index in skf.split(training[features], training[['new_Quality']]):
    indices.append([train_index, test_index])

iter_list = [0]
lr_list = [10, 5, 1, 0.5, 0.1, 0.05, 0.01]
depth_list = [4]
setting_list = []
for iteration in iter_list:
    for lr in lr_list:
        for depth in depth_list:
            setting_list.append((iteration, lr, depth))

stacking_results = np.zeros((len(setting_list), 7))
current_best_offline_score = -float('inf')
current_best_offline_logloss = -float('inf')
current_best_offline_accuracy = -float('inf')
i = 0
for setting in tqdm.tqdm(setting_list):
    iteration, lr, depth = setting
    #model = CatBoostClassifier(iterations=iteration, depth=depth, learning_rate=lr, silent=True, task_type='GPU', loss_function = 'MultiClass')
    model = LogisticRegression(C=lr, multi_class='multinomial', solver='lbfgs')
    train_predictions = np.zeros((6000, 4))
    for j in range(N):
        train_index = indices[j][0]
        test_index = indices[j][1]
        X_train = new_train_data.loc[train_index, column_list]
        y_train = training.loc[train_index, ['new_Quality']]
        X_test = new_train_data.loc[test_index, column_list]
        model.fit(X_train, y_train)
        train_predictions[test_index, :] += model.predict_proba(X_test)
    neg_log_loss = -log_loss(training['new_Quality'], train_predictions)
    accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predictions, 1))
    offline_score1 = []
    offline_score2 = []
    for j in range(100):
        scores = approx_score(train_predictions, training['new_Quality'].values, training['group_{}'.format(j)].values)
        offline_score1.append(scores[0])
        offline_score2.append(scores[1])
    stacking_results[i, :] = np.array([iteration, lr, depth, np.mean(offline_score1), np.mean(offline_score2), neg_log_loss, accuracy])
    if np.mean(offline_score2) > current_best_offline_score:
        current_best_offliene_score = np.mean(offline_score2)
        model.fit(new_train_data, training[['new_Quality']])
        submission_offline_score = model.predict_proba(new_test_data[column_list])
    if neg_log_loss > current_best_offline_logloss:
        current_best_offline_logloss = neg_log_loss
        model.fit(new_train_data, training[['new_Quality']])
        submission_offline_logloss = model.predict_proba(new_test_data[column_list])
    if accuracy > current_best_offline_accuracy:
        current_best_offline_accuracy = accuracy
        model.fit(new_train_data, training[['new_Quality']])
        submission_offline_accuracy = model.predict_proba(new_test_data[column_list])
    i += 1

stacking_results = pd.DataFrame(data = stacking_results, columns = ['iteration',
                                                                'learning rate',
                                                                'depth',
                                                                'validate_offline_score_mean',
                                                                'validate_offline_score_rounded',
                                                                'neg_log_loss',
                                                                'accuracy'])
stacking_results.to_pickle('./stacking/stacking_log.pkl')

get_prediction(testing, submission_offline_score, submit=True, name='Stacking_score')
get_prediction(testing, submission_offline_logloss, submit=True, name='Stacking_logloss')
get_prediction(testing, submission_offline_accuracy, submit=True, name='Stacking_accuracy')
    


