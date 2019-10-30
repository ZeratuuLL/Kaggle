[This is the link](https://www.datafountain.cn/competitions/351) to this contest. It's a Chinese page.

My solution is 153/2265.

Here is the description of the files:
  * Notebooks:
    + **CCF Contest EDA.ipynb** : EDA of the dataset, including some simple variable selection. Both logical and experimental
    + **CatBoost.ipynb** : Use an experiment to show that the models which are better on training set with out-of-fold prediction might not be better on test set. Questioning a good way to tell good models from bad models
    + **Categories.ipynb** : treat the features as categorical variables and treat labels as ordinal variables to train. Target Mean Encoding and Weight of Evidence Encoding, as well as PCA are included.
    + **Fusion.ipynb** : explains how we do model fusion for this contest
    + **submit.ipynb** : A notebook where the prediction will be given after simply setting some parameters
  * python source codes:
    + **all_scores.py** : a program especially tunes the hyperparameters for CatBoost with returning different measurements of out-of-fold prediction on training set. 
    + **fusion.py** : a program which does model fusion based on given models. Results will be save in *Fusion* folder
    + **my_GridSearch.py** : a grid search for hyperparameters wrote by my own. Similar to **parameter_tuning.py** but here features/preprocessings as well as hyperparameters were selected.
    + **parameter_tuning.py** : a program which tunes the hyperparameters for Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost. The search results will be saved in *Tuning* folder. BTW, the search range is not vast
    + **stacking.py** : similar to **fusion.py**, but here the we will try to improve performance by model stacking instead of model fusion. Results will be saved in *Stacking* folder
    + **transforms.py** : a source code file with all feature engineering I tried. 
    + **utils.py** : a source code file with some functions that are used by other source codes.
