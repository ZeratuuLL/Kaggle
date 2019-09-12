[This is the link](https://www.datafountain.cn/competitions/351) to this contest. It's a Chinese page.

Here is the description of the files:

  * **CCF Contest EDA.ipynb** : EDA of the dataset, including some simple variable selection. Both logical and experimental
  * **Categories.ipynb** : treat the features as categorical variables and treat labels as ordinal variables to train
  * **parameter_tuning.py** : a time consuming program which tunes the hyperparameters. Notice that the code cannot run at the same time. It seems that my machine refueses to do GridSearchCV for the second XGBoost model. So if you want to run it, please comment out some part of it.
