# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import pandas as pd

def kfold(dataframe, groups):
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    size = int(len(dataframe) / groups)
    total = size * groups
    features = [dataframe.iloc[i:i+size,1:] for i in range(0, total-1, size)]
    response = [dataframe.iloc[i:i+size,0] for i in range(0, total-1, size)]
    if total < len(dataframe):
        features[-1] = pd.concat([features[-1], dataframe.iloc[total:, 104:]])
        response[-1] = pd.concat([response[-1], dataframe.iloc[total:, 22]])
    return features[0], pd.concat(features[1:]), response[0], pd.concat(response[1:])