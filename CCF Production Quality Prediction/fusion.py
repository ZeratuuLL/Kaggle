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

lambd_list = [0, 0.1, 0.5, 1, 5, 10]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
                   #(RandomForestClassifier(n_estimators = 250, criterion = 'entropy', max_depth = 6, n_jobs=10), 'RF'),
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
        np.save('./fusion/'+name_list[-1]+'_train.npy', train_predict)
        np.save('./fusion/'+name_list[-1]+'_test.npy', test_predict)
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
    train_results.to_pickle('./fusion/fusion_train.pkl')

print('name list is {}'.format(name_list))
train_list = [np.load('./fusion/{}_train.npy'.format(name)) for name in name_list]
test_list = [np.load('./fusion/{}_test.npy'.format(name)) for name in name_list]

skf = StratifiedKFold(n_splits=N, shuffle=True)
indices = []
for train_index, test_index in skf.split(training[features], training[['new_Quality']]):
    indices.append([train_index, test_index])

fusion_results = np.zeros((len(lambd_list), 5))
print(fusion_results.shape)
current_best = 0
i = 0
for lambd in tqdm.tqdm(lambd_list):
    new_train = np.zeros((6000, 4))
    for k in range(N):
        train_index = indices[k][0]
        test_index = indices[k][1]
        alphas, _, _ = optimize([value[train_index, :] for value in train_list], L[train_index, :], 0.001, lambd)
        for j, alpha in enumerate(alphas):
            new_train[test_index, :] += alpha*train_list[j][test_index, :]
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
        lambd_best = lambd
        current_best = np.mean(offline_score2)
    i += 1

fusion_results = pd.DataFrame(data = fusion_results, columns = ['lambda',
                                                                'validate_offline_score_mean',
                                                                'validate_offline_score_rounded',
                                                                'neg_log_loss',
                                                                'accuracy'])
fusion_results.to_pickle('./fusion/fusion_log.pkl')

alpha_best, _, _ = optimize(train_list, L, 0.001, lambd_best)
submission = np.zeros((6000, 4))
for j, alpha in enumerate(alpha_best):
    submission += alpha*test_list[j]
get_prediction(testing, submission, submit=True, name='Fusion')
    


