from Merge import Merge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
import time
import numpy as np

Training = Merge(c1=0.00001, c2=0.01)

y = Training['revenue']
X = Training.drop(['revenue'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

t1 = time.time()
predictor0 = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
predictor0.fit(X_train, y_train)
y_test_prediction = predictor0.predict(X_test)
loss = np.mean((y_test_prediction-y_test)**2)
print('Train a simple regression tree with depth 10 requires {} sec. The loss is {}'.format(time.time()-t1, loss))

t1 = time.time()
predictor1 = DecisionTreeRegressor(max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
predictor1.fit(X_train, y_train)
y_test_prediction = predictor1.predict(X_test)
loss = np.mean((y_test_prediction-y_test)**2)
print('Train a simple regression tree with depth 20 requires {} sec. The loss is {}'.format(time.time()-t1, loss))

t1 = time.time()
predictor2 = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=2, random_state=42)
predictor2.fit(X_train, y_train)
y_test_prediction = predictor2.predict(X_test)
loss = np.mean((y_test_prediction-y_test)**2)
print('Train a random forest with 100 trees requires {}. The loss is {}'.format(time.time()-t1, loss))

t1 = time.time()
predictor3 = RandomForestRegressor(n_estimators=1000, max_depth=5, min_samples_split=10, min_samples_leaf=2, random_state=42)
predictor3.fit(X_train, y_train)
y_test_prediction = predictor3.predict(X_test)
loss = np.mean((y_test_prediction-y_test)**2)
print('Train a random forest with 100 trees requires {}. The loss is {}'.format(time.time()-t1, loss))
