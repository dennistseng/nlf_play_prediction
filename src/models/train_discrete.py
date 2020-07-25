# -*- coding: utf-8 -*-
"""
Clean Module for NFL Play Prediction Project
@author: Dennis Tseng

Part of the DSC 672 Capstone Final Project Class
Group 3: Dennis Tseng, Scott Elmore, Dongmin Sun

"""

#######################################################
## Load Required Libraries
#######################################################

import pandas as pd

# Some sklearn tools for preprocessing. 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Our algorithms, by from the easiest to the hardest to intepret.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, cv, Pool


# %%
#######################################################
## Load Dataset
#######################################################






# %%
#######################################################
## Pre-Processing
#######################################################

preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 
                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                   cat_features)])
