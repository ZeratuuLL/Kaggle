from transforms import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from multiprocessing import Pool
from sklearn.metrics import log_loss, accuracy_score
from sklearn.decomposition import PCA

pca = PCA()

def pre_processing(training, testing, type, num_features = -1):
    if type not in ['Original', 'PCA', 'TargetMean', 'WoE']:
        new_features = training.columns.values()
        train_all, test_all = training, testing
    if type == 'Original':
        new_features = ['Parameter'+str(i) for i in [5, 7, 8, 9, 10]]
        train_all, test_all = training.copy(), testing.copy()
        new_features = new_features[:num_features]
    if type == 'PCA':
        new_features = ['Parameter'+str(i) for i in range(5, 11)]
        train_all, test_all = training.copy(), testing.copy()
        new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
        train_all[new_features] = new_values[:6000, :].copy()
        test_all[new_features] = new_values[6000:, :].copy()
        new_features = new_features[:num_features]
    if type == 'TargetMean':
        features = ["Parameter5","Parameter6","Parameter7","Parameter8","Parameter9","Parameter10"]
        train_all, test_all, new_features = get_all_encoding(training, testing, features)
        new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
        train_all[new_features] = new_values[:6000, :].copy()
        test_all[new_features] = new_values[6000:, :].copy()
        new_features = new_features[:num_features]
    if type == 'WoE':
        features = ["Parameter5","Parameter6","Parameter7","Parameter8","Parameter9","Parameter10"]
        train_all, test_all, new_features = get_all_WoE(training, testing, features)
        new_values = pca.fit_transform(pd.concat([train_all[new_features], test_all[new_features]]))
        train_all[new_features] = new_values[:6000, :].copy()
        test_all[new_features] = new_values[6000:, :].copy()
        new_features = new_features[:num_features]
    return train_all, test_all, new_features
    
def gradient(A_list, L, alpha, lambd, regularization='KL'):
    '''
    Inputs:
    =======
    A_list : list of numpy arrays containing prediction probabilities
    L : true 0-1 numpy array
    alpha : length K vector as fusion weights
    lambda : value of lambda, the strength of regularization term
    regularization : string for choice of regularization. default 'KL'. Can be 'L2'. If neither, 'KL' would be use
    
    Return:
    =======
    grads : lenght K-1 vector, which is the gradient of first K-1 elements of alpha
    '''
    if not regularization in ['KL', 'L2']:
        regularization = 'KL'
    
    N = A_list[0].shape[0]
    K = len(A_list)
    A = np.zeros(A_list[0].shape)
    for k in range(K):
        A += alpha[k] * A_list[k]
    grad = np.zeros(K-1)
    for i in range(K-1):
        grad[i] = np.mean(np.sum((A_list[i] - A_list[-1])/(A+1e-6)*L, axis=1))#partial L/partial alpha_i
    for i in range(K-1):
        if regularization=='KL':
            grad[i] += lambd/K * (1/alpha[i]-1/alpha[-1])
        else:
            grad[i] -= 2*lambd*(alpha[i] - alpha[-1])
    
    return grad

def target_function(A_list, alpha, L, lambd, regularization):
    K = len(A_list)
    A = np.zeros(A_list[0].shape)
    for i in range(K):
        A += A_list[i] * alpha[i]
    likelihood = np.mean(np.sum(np.log(A)*L, axis=1))
    if regularization == 'KL':
        reg = lambd/K * np.sum(np.log(K*alpha))
    else:
        reg = -lambd * np.sum((lambd - 1/K)**2)
    return likelihood + reg

def optimize(A_list, L, step_size, lambd=0, regularization='KL', max_iter=10000, eps=1e-5):
    K = len(A_list)
    alpha = np.ones(K)/K
    counter = 0
    log = []
    if not regularization in ['KL', 'L2']:
        regularization = 'KL'
    while counter<max_iter:
        log.append(target_function(A_list, alpha, L, lambd, regularization))
        grads = np.zeros(K-1)
        grads = gradient(A_list, L, alpha, lambd, regularization)
        if np.mean(grads**2)/np.mean(alpha[:(K-1)]**2) < eps:
            return alpha, log, counter
        alpha[:(K-1)] += step_size * grads
        alpha[-1] = 1 - np.sum(alpha[:(K-1)])
        counter += 1
    return alpha, log, counter

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

def get_prediction(testing, prob_predict, submit=False, name='submission'):
    testing['Fail ratio'] = 0
    testing['Pass ratio'] = 0
    testing['Good ratio'] = 0
    testing['Excellent ratio'] = 0
    testing[['Fail ratio', 'Pass ratio', 'Good ratio', 'Excellent ratio']] = prob_predict
    submission = testing.groupby(['Group'], as_index=False).mean()
    submission = submission[['Group', 'Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']]
    if submit:
        submission.to_csv('{}.csv'.format(name), index=False)
    matrix1 = submission[['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']].values.copy()
    matrix2 = normalize(matrix1)
    submission[['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']] = matrix2.copy()
    if submit:
        submission.to_csv('./Submissions/{}_rounded.csv'.format(name), index=False)
    return matrix1, matrix2

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

def CV(model, train, test, indices, feature_list, target, predict_on_test=False):
    '''
    This function takes a model as input and return the N-Fold CV predictions
    
    Other Params:
    =============
    train : training set
    test : test set
    indices : indices for the N-folds
    feature_list : list of features for training
    target : name of target column
    prediction_on_test : whether should fit the training set and predict the test set
    '''
    N = len(indices)
    train_predict = np.zeros((train.shape[0], 4))
    test_predict = np.zeros((test.shape[0], 4))
    for j in range(N):
        train_index = indices[j][0]
        test_index = indices[j][1]
        X_train = train.loc[train_index, feature_list]
        y_train = train.loc[train_index, [target]]
        X_test = train.loc[test_index, feature_list]
        model.fit(X_train, y_train)
        train_predict[test_index, :] += model.predict_proba(train.loc[test_index, feature_list])
    if predict_on_test:
        model.fit(train[feature_list], train[[target]])
        test_predict = model.predict_proba(test[feature_list])
    return train_predict, test_predict