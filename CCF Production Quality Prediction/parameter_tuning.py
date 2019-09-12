import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

text_file=open('training.log','w')
NJOBS = 3
N_ESTIMATOR = [100, 200, 300, 400, 500, 800, 1000]
MAX_DEPTH = [5, 6, 7, 8, 9, 10]
LEARNING_RATE = [0.05, 0.01, 0.005]
ITERATIONS = [1500, 1750, 2000, 2250, 2500]
BOOSTING = ['gbdt', 'goss']

training = pd.read_csv('first_round_training_data.csv')
testing = pd.read_csv('first_round_testing_data.csv')

code = {'Pass':1, 'Good':2, 'Excellent':3, 'Fail':0}
training['new_Quality'] = training['Quality_label'].apply(lambda x : code[x])

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

light_estimator = LGBMClassifier(objective='multiclass', n_jobs=3)
GBT_estimator = GradientBoostingClassifier()
randomforest_estimator = RandomForestClassifier()
xgb_estimator = XGBClassifier(objective='multi:softmax', verbosity=0, verbose_eval=False)
cat_estimator = CatBoostClassifier(loss_function='MultiClass', silent=True)

common_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=302)

features = ['Parameter' + str(i) for i in [5, 7, 8, 9, 10]]

def my_print(a1, a2, a3, a4, a5, a6, a7):
    text_file.write('The best test accuracy is {}\n'.format(a1))
    text_file.write('The best test neg log loss is {}\n'.format(a2))
    text_file.write('The best parameters are {}\n'.format(a3))
    text_file.write('The top five results have training accuracy {}\n'.format(a4))
    text_file.write('The top five results have test accuracy {}\n'.format(a5))
    text_file.write('The top five results have training neg log loss {}\n'.format(a6))
    text_file.write('The top five results have test neg log loss {}\n'.format(a7))

def submission(name):
    prediction = testing.groupby(['Group'],as_index=False)['prob_Excellent','prob_Good','prob_Pass','prob_Fail'].mean()
    prediction.columns = ['Group','Excellent ratio','Good ratio','Pass ratio','Fail ratio']
    prediction.to_csv(name,index=False)

def randomforest_prediction(params):
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    criterion = params['criterion']
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
    clf.fit(training[features], training['new_Quality'])
    probs = clf.predict_proba(testing[features])
    testing['prob_Fail'] = 0
    testing['prob_Pass'] = 0
    testing['prob_Good'] = 0
    testing['prob_Excellent'] = 0
    testing.loc[:,['prob_Fail','prob_Pass','prob_Good','prob_Excellent']] = probs

def gradientboostingtree_prediction(params):
    learning_rate = params['learning_rate']
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(training[features], training['new_Quality'])
    probs = clf.predict_proba(testing[features])
    testing['prob_Fail'] = 0
    testing['prob_Pass'] = 0
    testing['prob_Good'] = 0
    testing['prob_Excellent'] = 0
    testing.loc[:,['prob_Fail','prob_Pass','prob_Good','prob_Excellent']] = probs

def xgb_prediction(params):
    learning_rate = params['learning_rate']
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    clf = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(training[features], training['new_Quality'])
    probs = clf.predict_proba(testing[features])
    testing['prob_Fail'] = 0
    testing['prob_Pass'] = 0
    testing['prob_Good'] = 0
    testing['prob_Excellent'] = 0
    testing.loc[:,['prob_Fail','prob_Pass','prob_Good','prob_Excellent']] = probs

def light_prediction(params):
    learning_rate = params['learning_rate']
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    boosting_type = params['boosting_type']
    clf = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, boosting_type=boosting_type, objective='multiclass')
    clf.fit(training[features], training['new_Quality'])
    probs = clf.predict_proba(testing[features])
    testing['prob_Fail'] = 0
    testing['prob_Pass'] = 0
    testing['prob_Good'] = 0
    testing['prob_Excellent'] = 0
    testing.loc[:,['prob_Fail','prob_Pass','prob_Good','prob_Excellent']] = probs

def cat_prediction(params):
    learning_rate = params['learning_rate']
    iterations = params['iterations']
    depth = params['depth']
    clf = CatBoostClassifier(learning_rate=learning_rate, iterations = iterations, depth=depth, loss_function='MultiClass', silent=True)
    clf.fit(training[features], training['new_Quality'])
    probs = clf.predict_proba(testing[features])
    testing['prob_Fail'] = 0
    testing['prob_Pass'] = 0
    testing['prob_Good'] = 0
    testing['prob_Excellent'] = 0
    testing.loc[:,['prob_Fail','prob_Pass','prob_Good','prob_Excellent']] = probs

text_file.write('Random Forest Accuracy\n')
parameters = {'n_estimators':N_ESTIMATOR, 
              'max_depth':MAX_DEPTH, 
              'criterion':['gini', 'entropy']}
random_forest_cv1 = GridSearchCV(cv=10,
                                estimator=randomforest_estimator,
                                param_grid=parameters,
                                scoring=['accuracy', 'neg_log_loss'],
                                refit='accuracy',
                                n_jobs=NJOBS)
random_forest_cv1.fit(training[features], training['new_Quality'])
index = random_forest_cv1.best_index_
my_print(random_forest_cv1.cv_results_['mean_test_accuracy'][index], 
 random_forest_cv1.cv_results_['mean_test_neg_log_loss'][index], 
 random_forest_cv1.best_params_,
 [random_forest_cv1.cv_results_['mean_train_accuracy'][random_forest_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [random_forest_cv1.cv_results_['mean_test_accuracy'][random_forest_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [random_forest_cv1.cv_results_['mean_train_neg_log_loss'][random_forest_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [random_forest_cv1.cv_results_['mean_test_neg_log_loss'][random_forest_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
randomforest_prediction(random_forest_cv1.best_params_)
submission('RF_Accuracy_NO.csv')

text_file.write('\n\nRandom Forest Log Loss\n')
parameters = {'n_estimators':N_ESTIMATOR, 
              'max_depth':MAX_DEPTH, 
              'criterion':['gini', 'entropy']}
random_forest_cv2 = GridSearchCV(cv=10,
                                estimator=randomforest_estimator,
                                param_grid=parameters,
                                scoring=['accuracy', 'neg_log_loss'],
                                refit='neg_log_loss',
                                n_jobs=NJOBS)
random_forest_cv2.fit(training[features], training['new_Quality'])
index = random_forest_cv2.best_index_
my_print(random_forest_cv2.cv_results_['mean_test_accuracy'][index], 
 random_forest_cv2.cv_results_['mean_test_neg_log_loss'][index], 
 random_forest_cv2.best_params_,
 [random_forest_cv2.cv_results_['mean_train_accuracy'][random_forest_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)],
 [random_forest_cv2.cv_results_['mean_test_accuracy'][random_forest_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)],
 [random_forest_cv2.cv_results_['mean_train_neg_log_loss'][random_forest_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)],
 [random_forest_cv2.cv_results_['mean_test_neg_log_loss'][random_forest_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)])
randomforest_prediction(random_forest_cv2.best_params_)
submission('RF_LogLoss_NO.csv') 

text_file.write('\n\nGBT Accuracy\n')
parameters = {'learning_rate':LEARNING_RATE,
              'n_estimators':N_ESTIMATOR,
              'max_depth':MAX_DEPTH}
GBT_cv1 = GridSearchCV(cv=10,
                      estimator=GBT_estimator,
                      param_grid=parameters,
                      scoring=['accuracy', 'neg_log_loss'],
                      refit='accuracy',
                      n_jobs=NJOBS)
GBT_cv1.fit(training[features], training['new_Quality'])
index = GBT_cv1.best_index_
my_print(GBT_cv1.cv_results_['mean_test_accuracy'][index], 
 GBT_cv1.cv_results_['mean_test_neg_log_loss'][index], 
 GBT_cv1.best_params_,
 [GBT_cv1.cv_results_['mean_train_accuracy'][GBT_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [GBT_cv1.cv_results_['mean_test_accuracy'][GBT_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [GBT_cv1.cv_results_['mean_train_neg_log_loss'][GBT_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [GBT_cv1.cv_results_['mean_test_neg_log_loss'][GBT_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
gradientboostingtree_prediction(GBT_cv1.best_params_)
submission('GBT_Accuracy_NO.csv') 

text_file.write('\n\nGBT Log Loss\n')
parameters = {'learning_rate':LEARNING_RATE,
              'n_estimators':N_ESTIMATOR,
              'max_depth':MAX_DEPTH}
GBT_cv2 = GridSearchCV(cv=10,
                      estimator=GBT_estimator,
                      param_grid=parameters,
                      scoring=['accuracy', 'neg_log_loss'],
                      refit='neg_log_loss',
                      n_jobs=NJOBS)
GBT_cv2.fit(training[features], training['new_Quality'])
index = GBT_cv2.best_index_
my_print(GBT_cv2.cv_results_['mean_test_accuracy'][index], 
 GBT_cv2.cv_results_['mean_test_neg_log_loss'][index], 
 GBT_cv2.best_params_,
 [GBT_cv2.cv_results_['mean_train_accuracy'][GBT_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)],
 [GBT_cv2.cv_results_['mean_test_accuracy'][GBT_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)],
 [GBT_cv2.cv_results_['mean_train_neg_log_loss'][GBT_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)],
 [GBT_cv2.cv_results_['mean_test_neg_log_loss'][GBT_cv2.cv_results_['rank_test_neg_log_loss']==i] for i in range(1, 6)])
gradientboostingtree_prediction(GBT_cv2.best_params_)
submission('GBT_LogLoss_NO.csv')

text_file.write('\n\nXGB Accuracy\n')
parameters = {'learning_rate':LEARNING_RATE,
              'n_estimators':N_ESTIMATOR,
              'max_depth':MAX_DEPTH}
xgb_cv1 = GridSearchCV(cv=common_cv,
                       estimator=xgb_estimator,
                       param_grid=parameters,
                       scoring=['accuracy', 'neg_log_loss'],
                       refit='accuracy',
                       verbose=0,
                       n_jobs=NJOBS)
xgb_cv1.fit(training[features], training['new_Quality'])
index = xgb_cv1.best_index_
my_print(xgb_cv1.cv_results_['mean_test_accuracy'][index], 
 xgb_cv1.cv_results_['mean_test_neg_log_loss'][index], 
 xgb_cv1.best_params_,
 [xgb_cv1.cv_results_['mean_train_accuracy'][xgb_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [xgb_cv1.cv_results_['mean_test_accuracy'][xgb_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [xgb_cv1.cv_results_['mean_train_neg_log_loss'][xgb_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [xgb_cv1.cv_results_['mean_test_neg_log_loss'][xgb_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
xgb_prediction(xgb_cv1.best_params_)
submission('XGB_Auucracy_No.csv')

text_file.write('\n\nXGB Log Loss\n')
parameters = {'learning_rate':LEARNING_RATE,
              'n_estimators':N_ESTIMATOR,
              'max_depth':MAX_DEPTH}
xgb_cv2 = GridSearchCV(cv=common_cv,
                       estimator=xgb_estimator,
                       param_grid=parameters,
                       scoring=['accuracy', 'neg_log_loss'],
                       refit='neg_log_loss',
                       verbose=0,
                       n_jobs=NJOBS)
xgb_cv2.fit(training[features], training['new_Quality'])
index = xgb_cv2.best_index_
my_print(xgb_cv2.cv_results_['mean_test_accuracy'][index], 
 xgb_cv2.cv_results_['mean_test_neg_log_loss'][index], 
 xgb_cv2.best_params_,
 [xgb_cv2.cv_results_['mean_train_accuracy'][xgb_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [xgb_cv2.cv_results_['mean_test_accuracy'][xgb_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [xgb_cv2.cv_results_['mean_train_neg_log_loss'][xgb_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [xgb_cv2.cv_results_['mean_test_neg_log_loss'][xgb_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
xgb_prediction(xgb_cv2.best_params_)
submission('XGB_LogLoss_No.csv')

text_file.write('\n\nCAT Accuracy\n')
parameters = {'learning_rate':LEARNING_RATE,
              'iterations':ITERATIONS,
              'depth':MAX_DEPTH}
cat_cv1 = GridSearchCV(cv=common_cv,
                       estimator=cat_estimator,
                       param_grid=parameters,
                       scoring=['accuracy', 'neg_log_loss'],
                       refit='accuracy',
                       verbose=0,
                       n_jobs=NJOBS)
cat_cv1.fit(training[features], training['new_Quality'])
index = cat_cv1.best_index_
my_print(cat_cv1.cv_results_['mean_test_accuracy'][index], 
 cat_cv1.cv_results_['mean_test_neg_log_loss'][index], 
 cat_cv1.best_params_,
 [cat_cv1.cv_results_['mean_train_accuracy'][cat_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [cat_cv1.cv_results_['mean_test_accuracy'][cat_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [cat_cv1.cv_results_['mean_train_neg_log_loss'][cat_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [cat_cv1.cv_results_['mean_test_neg_log_loss'][cat_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
cat_prediction(cat_cv1.best_params_)
submission('CAT_Accuracy_No.csv')

text_file.write('\n\nCAT LogLoss\n')
parameters = {'learning_rate':LEARNING_RATE,
              'iterations':ITERATIONS,
              'depth':MAX_DEPTH}
cat_cv2 = GridSearchCV(cv=common_cv,
                       estimator=cat_estimator,
                       param_grid=parameters,
                       scoring=['accuracy', 'neg_log_loss'],
                       refit='neg_log_loss',
                       verbose=0,
                       n_jobs=NJOBS)
cat_cv2.fit(training[features], training['new_Quality'])
index = cat_cv2.best_index_
my_print(cat_cv2.cv_results_['mean_test_accuracy'][index], 
 cat_cv2.cv_results_['mean_test_neg_log_loss'][index], 
 cat_cv2.best_params_,
 [cat_cv2.cv_results_['mean_train_accuracy'][cat_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [cat_cv2.cv_results_['mean_test_accuracy'][cat_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [cat_cv2.cv_results_['mean_train_neg_log_loss'][cat_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [cat_cv2.cv_results_['mean_test_neg_log_loss'][cat_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
cat_prediction(cat_cv2.best_params_)
submission('CAT_LogLoss_No.csv')

text_file.write('\n\nLight Accuracy\n')
parameters = {'learning_rate':LEARNING_RATE,
              'n_estimators':ITERATIONS,
              'max_depth':MAX_DEPTH,
              'boosting_type':BOOSTING}
light_cv1 = GridSearchCV(cv=common_cv,
                       estimator=light_estimator,
                       param_grid=parameters,
                       scoring=['accuracy', 'neg_log_loss'],
                       refit='accuracy',
                       verbose=0,
                       n_jobs=NJOBS)
light_cv1.fit(training[features], training['new_Quality'])
index = light_cv1.best_index_
my_print(light_cv1.cv_results_['mean_test_accuracy'][index], 
 light_cv1.cv_results_['mean_test_neg_log_loss'][index], 
 light_cv1.best_params_,
 [light_cv1.cv_results_['mean_train_accuracy'][clight_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [clight_cv1.cv_results_['mean_test_accuracy'][light_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [light_cv1.cv_results_['mean_train_neg_log_loss'][light_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [light_cv1.cv_results_['mean_test_neg_log_loss'][light_cv1.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
light_prediction(lightlight_cv1.best_params_)
submission('LIGHT_Accuracy_No.csv')

text_file.write('\n\nLight Accuracy\n')
parameters = {'learning_rate':LEARNING_RATE,
              'n_estimators':ITERATIONS,
              'max_depth':MAX_DEPTH,
              'boosting_type':BOOSTING}
light_cv2 = GridSearchCV(cv=common_cv,
                       estimator=light_estimator,
                       param_grid=parameters,
                       scoring=['accuracy', 'neg_log_loss'],
                       refit='neg_log_loss',
                       verbose=0,
                       n_jobs=NJOBS)
light_cv2.fit(training[features], training['new_Quality'])
index = light_cv2.best_index_
my_print(light_cv2.cv_results_['mean_test_accuracy'][index], 
 light_cv2.cv_results_['mean_test_neg_log_loss'][index], 
 light_cv2.best_params_,
 [light_cv2.cv_results_['mean_train_accuracy'][clight_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [clight_cv2.cv_results_['mean_test_accuracy'][light_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [light_cv2.cv_results_['mean_train_neg_log_loss'][light_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)],
 [light_cv2.cv_results_['mean_test_neg_log_loss'][light_cv2.cv_results_['rank_test_accuracy']==i] for i in range(1, 6)])
light_prediction(lightlight_cv2.best_params_)
submission('LIGHT_LogLoss_No.csv')


text_file.close()
