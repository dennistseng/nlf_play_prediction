# -*- coding: utf-8 -*-
"""
Train Module for NFL Play Prediction Project
@author: Dennis Tseng

Part of the DSC 672 Capstone Final Project Class
Group 3: Dennis Tseng, Scott Elmore, Dongmin Sun

'Branched' from Casey Bennett's sklearn template. Modified by Dennis Tseng
to allow for the use of pipelines and imblearn

"""

#############################################################################
#
# Load Required Libraries
#
#####################

# Standard packages
import sys
import csv
import math
import numpy as np
import pandas as pd
from operator import itemgetter
import time
from collections import Counter

# Pipeline, if not using imblearn comment that out and use sklearn's pipeline instead
#from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline, Pipeline

# Some sklearn tools for preprocessing, feature selection and etc
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, scale, MinMaxScaler
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn.decomposition import PCA

# Imbalanced Datasets
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, cv, Pool
from lightgbm.sklearn import LGBMClassifier

# Model Evaluation
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer

# Model Exporting
#from sklearn.externals import joblib

# Model Interpretability
import shap
import lime

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


# %% 
#############################################################################
#
# Global parameters
#
#####################

imb_class=0                                         #Control switch for type of sampling to deal with imbalanced class (0=None, 1=SMOTE, 2=NearMiss)
cross_val=1                                         #Control Switch for CV
norm_features=1                                     #Normalize features switch
pca = 0
feat_select=0                                       #Control Switch for Feature Selection
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=0                                        #Start column of features
k_cnt= 30                                           #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
param_tuning = 0                                    #Turn on model parameter tuning

#Set global model parameters
rand_st=0                                           #Set Random State variable for randomizing splits on runs

# %%
#############################################################################
#
# Load Data
#
#####################

data = pd.read_csv("../../data/clean/model_plays.csv", low_memory = False)
data['target'].value_counts()


# Test for run and pass predictors. Comment out if we don't want this
data = data[((data['target'] == 'run') |  (data['target'] == 'pass'))]

# Separate Target from dataset

#data['target'] = data['target'].astype('category')
#target_cat_label = dict(enumerate(data.target.categories))

# Change categorical variables to numerical
data['target'] = data['target'].astype('category').cat.codes
# data['target'] = data['target'].map(target_cat_label)

target = data['target']
del data['target']

# Split prior to any normalization/sampling. Stratify is set understanding that we have imbalanced classes and want to preserve that in the split
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20, stratify = target)

# %%
#############################################################################
#
# Create pipeline lists 
#
##########################################

# sklearn
knn_pipe = []
dt_pipe = []
rf_pipe = []
ab_pipe = []
gb_pipe = []
svm_pipe = []
mlp_pipe = []

skpipes = [knn_pipe, dt_pipe, rf_pipe, ab_pipe, gb_pipe, svm_pipe, mlp_pipe]

# %%
#############################################################################
#
# Pre-processing
#
##########################################

# In the future ColumnTransformers can be utilized... actually it will be once this pipeline is finished
'''
preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 
                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                   cat_features)])
'''

# Normalizing using Standard Scaler
if norm_features == 1:
    for i in range(0, len(skpipes)):
        skpipes[i].append(('scalar'+str(i),StandardScaler()))

     
# Dealing with imbalanced classes
if imb_class == 0:
    pass
elif imb_class == 1:
    # Oversample with SMOTE
    for i in range(0, len(skpipes)):
        skpipes[i].append(('smote'+str(i), SMOTE(random_state = rand_st)))
        
elif imb_class == 2:
    # Undersample using NearMiss
    for i in range(0, len(skpipes)):
        skpipes[i].append(('smote'+str(i), NearMiss(version=1)))
    

# %%
#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('--LOW VARIANCE FILTER ON--', '\n')
    
    for i in range(0, len(skpipes)):
        skpipes[i].append(('variance_threshold'+str(i), VarianceThreshold(threshold = 0.5)))   


#Feature Selection
if feat_select==1:
    print('--FEATURE SELECTION ON--', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None, class_weight="balanced")
        cv_rfc = RFECV(clf, n_features_to_select=k_cnt, step=.1, cv = 5, scoring = 'roc_auc')
            
        for i in range(0, len(skpipes)):
            skpipes[i].append(('rfe_rf'+str(i), cv_rfc))   
        
'''
    if fs_type==2:
        #Wrapper Select via model
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
        sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
        print ('Wrapper Select - Random Forest: ')
        
        fit_mod=sel.fit(data_np, target_np)    
        sel_idx=fit_mod.get_support()

    if fs_type==3:                                                                ######Only work if the Target is binned###########
        #Univariate Feature Selection - Chi-squared
        #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
        sel=SelectKBest(chi2, k=k_cnt)
        fit_mod=sel.fit(data_np, target_np)                                         #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
        print ('Univariate Feature Selection - Chi2: ')
        sel_idx=fit_mod.get_support()

        #Print ranked variables out sorted
        temp=[]
        scores=fit_mod.scores_
        for i in range(feat_start, len(data_np.columns)):            
            temp.append([data_np.columns[i], float(scores[i-feat_start])])

        print('Ranked Features')
        temp_sort=sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')
'''


# %%
#############################################################################
#
# Dimensionality Reduction
#
##########################################


# PCA Dimensionality Reduction
if pca == 1:
    for i in range(0, len(skpipes)):
        skpipes[i].append(('pca'+str(i), PCA(n_components = 0.95)))    
        


# %%
#############################################################################
#
# Add Classifiers
#
##########################################

'''
# Logistic Regression
lr_pipe = Pipeline([('scaler', StandardScaler()),
                     ('classifier', LogisticRegression())
                     ])
'''

# List of Classifiers
classifiers = [('knn_classifier', KNeighborsClassifier()),
               
               # Tree-based methods
               ('dt_classifier', DecisionTreeClassifier()), 
               ('rf_classifier', RandomForestClassifier()),
               ('ab_classifier', AdaBoostClassifier()),
               ('gb_classifier', GradientBoostingClassifier()),
               
               # Support Vector Machines
               ('sv_classifier', SVC()),
               
               # Neural Networks
               ('nn_classifier', MLPClassifier())
               ]

for i, c in enumerate(classifiers):
    skpipes[i].append(c) 

# %%
#############################################################################
#
# Create Pipelines
#
##########################################

'''
# Logistic Regression
lr_pipe = Pipeline([('scaler', StandardScaler()),
                     ('classifier', LogisticRegression())
                     ])
'''

# KNN
knn_pipe = Pipeline(skpipes[0])


# Tree Methods
dt_pipe = Pipeline(skpipes[1])
rf_pipe = Pipeline(skpipes[2])
ab_pipe = Pipeline(skpipes[3])
gb_pipe = Pipeline(skpipes[4])

# SVMs
svm_pipe = Pipeline(skpipes[5])

# Neural Networks
mlp_pipe = Pipeline(skpipes[6])


# List of all classifier pipelines
pipelines = [knn_pipe, dt_pipe, rf_pipe, ab_pipe, gb_pipe, svm_pipe, mlp_pipe]
best_accuracy=0.0
best_classifier=0
best_pipeline=""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {1: 'Decision Tree', 2: 'RandomForest', 3: 'AdaBoost', 4:'GradientBoostedTrees', 5:'svm', 6:'feedforwardnn'}


gb_pipe.fit(data_train, target_train)
print(gb_pipe.score(data_test,target_test))


# %% 
#############################################################################
#
# Hyperparameter Tuning
#
##########################################

best_params = []

if param_tuning == 1:
    
    
    dt_grid = {'criterion': ['gini'],
               'max_features': ['auto', 'sqrt', 'log2', None],
               'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
               'min_samples_split': [2, 5, 10, 20, 25, 50, 100],
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
               'class_weight': ['balanced', 'balanced_subsample', None]}
    
    
    
    grid_param = [{'selector__k': [5, 10, 20, 30]},
                    {'classifier': [LogisticRegression(solver='lbfgs')],
                     'classifier__C': [0.01, 0.1, 1.0]},
                    {'classifier': [RandomForestClassifier(n_estimators=100)],
                     'classifier__max_depth': [5, 10, None]},
                    {'classifier': [KNeighborsClassifier()],
                     'classifier__n_neighbors': [3, 7, 11],
                     'classifier__weights': ['uniform', 'distance']}]
    
    
    for pipe in skpipes:
        gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) 
        best_model = gridsearch.fit(data_train, target_train)
        best_model.score(data_train,target_test)
    
    
    
    
    
    clf = GridSearchCV(pipe, grid_param, cv=10, verbose=0)
    clf = clf.fit(data_train, target_train)
    
    clf.best_estimator_
    clf.best_score_