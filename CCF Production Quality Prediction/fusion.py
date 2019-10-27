from utils import *
from transforms import *
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import log_loss, accuracy_score
from catboost import CatBoostClassifier
import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
lambd_list = [0, 0.1, 0.5, 1, 5, 10]

training = pd.read_csv('first_round_training_data.csv')
testing = pd.read_csv('first_round_testing_data.csv')
features = ["Parameter5","Parameter6","Parameter7","Parameter8","Parameter9","Parameter10"]

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

# this is the list of candidates
candidates_list = [('Original', 5, 10, 0.01, 10),
                   ('PCA', 6, 10, 0.01, 10)]
name_list = []

if len(name_list) == 0 :
    def train(params):
        '''
        params are parameters to train the model. It should be a tuple of (train_type, num_features, depth, lr, l2)
        '''
        train_type, num_features, depth, learning_rate, l2_leaf_reg = params
        train_all, test_all, new_features = pre_processing(training, testing, train_type, num_features)
        model = CatBoostClassifier(iterations = 3000, 
                                   depth = depth, 
                                   learning_rate = learning_rate, 
                                   silent = True, 
                                   loss_function = 'MultiClass', 
                                   task_type = 'GPU',
                                   l2_leaf_reg = l2_leaf_reg,
                                   od_type = 'Iter',
                                   od_wait = 100)
        train_predict = np.zeros((6000, 4))
        test_predict = np.zeros((6000, 4))
        for j in range(N):
            train_index = indices[j][0]
            test_index = indices[j][1]
            X_train = train_all.loc[train_index, new_features]
            y_train = train_all.loc[train_index, ['new_Quality']]
            X_test = train_all.loc[test_index, new_features]
            y_test = train_all.loc[test_index, ['new_Quality']]
            model.fit(X_train, y_train, eval_set=(X_test, y_test))
            train_predict[test_index, :] += model.predict_proba(train_all.loc[test_index, new_features])
            test_predict += model.predict_proba(test_all[new_features])
        test_predict /= N
        neg_log_loss = -log_loss(training['new_Quality'], train_predict)
        accuracy = accuracy_score(training['new_Quality'], np.argmax(train_predict, 1))
        name_list.append('{}_{}'.format(train_type, len(new_features)))
        np.save(name_list[-1]+'_train.npy', train_predict)
        np.save(name_list[-1]+'_test.npy', test_predict)
        offline_score1 = []
        offline_score2 = []
        for j in range(100):
            scores = approx_score(train_predict, training['new_Quality'].values, training['group_{}'.format(j)].values)
            offline_score1.append(scores[0])
            offline_score2.append(scores[1])
        return type_to_num[train_type], len(new_features), depth, learning_rate, l2_leaf_reg, np.mean(offline_score1), np.mean(offline_score2), neg_log_loss, accuracy, 
    
    train_results = []
    for candidate in tqdm.tqdm(candidates_list):
        train_results.append(train(candidate))
    train_results = np.array(train_results)
    train_results = pd.DataFrame(data = train_results, columns = ['train_type',
                                                                  'num_features',
                                                                  'depth',
                                                                  'learning_rate',
                                                                  'l2_leaf_reg',
                                                                  'validate_offline_score_mean',
                                                                  'validate_offline_score_rounded',
                                                                  'neg_log_loss',
                                                                  'accuracy'])
    train_results['train_type'] = train_results['train_type'].apply(lambda x : num_to_type[x])
    train_results.to_pickle('fusion_train.pkl')

print('name list is {}'.format(name_list))
train_list = [np.load('{}_train.npy'.format(name)) for name in name_list]
test_list = [np.load('{}_test.npy'.format(name)) for name in name_list]

fusion_results = np.zeros((len(lambd_list), 5))
current_best = 0
i = 0
for lambd in tqdm.tqdm(lambd_list):
    alphas, _, _ = optimize(train_list, L, 0.01, lambd)
    new_train = np.zeros((6000, 4))
    for j, alpha in enumerate(alphas):
        new_train += alpha*train_list[j]
    neg_log_loss = -log_loss(training['new_Quality'], new_train)
    accuracy = accuracy_score(training['new_Quality'], np.argmax(new_train, 1))
    offline_score1 = []
    offline_score2 = []
    for j in range(100):
        scores = approx_score(new_train, training['new_Quality'].values, training['group_{}'.format(j)].values)
        offline_score1.append(scores[0])
        offline_score2.append(scores[1])
    fusion_results[i, :] = np.array([lambd, np.mean(offline_score1), np.mean(offline_score2), neg_log_loss, accuracy])
    if np.mean(offline_score2)>current_best:
        alpha_best = alphas
        current_best = np.mean(offline_score2)
    i += 1

fusion_results = pd.DataFrame(data = fusion_results, columns = ['lambda',
                                                                'validate_offline_score_mean',
                                                                'validate_offline_score_rounded',
                                                                'neg_log_loss',
                                                                'accuracy'])
fusion_results.to_pickle('fusion_log.pkl')

submission = np.zeros((6000, 4))
for j, alpha in enumerate(alpha_best):
    submission += alpha*test_list[j]
get_prediction(testing, submission, submit=True, name='Fusion')
    


