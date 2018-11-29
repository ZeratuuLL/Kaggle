The idea is to use two sets of variables. The first set contains variables that are already in the dataset, or can be easily extracted, like month, year, country, subcontinent etc. See the full list below. The other set contains some hand-designed features, which are used to estimate (or represent) a person's shopping habit ignoring the month factor. These variables can be directly estimated from data but most of the data are negative samples. So instead of an unbiased estimate we use the bayesian posterior MLE. The prior is obtained by empericial bayesian method. The goal of these hand-designed features is to avoid high dimensional categorical variables, which has to be transformed into saprse matrix.

List of first set (directly from dataset):
Year, month

List of second set (hand-designed features):
Distribution over contents, country, subcontinent, continent's shopping willingness and shopping power.
