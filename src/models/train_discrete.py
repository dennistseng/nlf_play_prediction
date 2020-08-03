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
from imblearn.under_sampling import NearMiss, RandomUnderSampler
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
import pickle

# Model Interpretability
import shap
import lime

# ML Flow
import mlflow.sklearn


#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


# %% 
#############################################################################
#
# Global parameters
#
#####################

imb_class=0                                         #Control switch for type of sampling to deal with imbalanced class (0=None, 1=SMOTE, 2=NearMiss, 3=RandomUnderSampler)
cross_val=1                                         #Control Switch for CV
norm_features=1                                     #Normalize features switch
pca = 0
feat_select=0                                       #Control Switch for Feature Selection
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=1                                         #Control switch for low variance filter on features
k_cnt= 20                                           #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
param_tuning = 1                                    #Turn on model parameter tuning
exhaustive_search = 1                               #Turn on if you want exhaustive grid search. Otherwise, it will default to RandomizedSearchCV

#Set global model parameters
rand_st=0                                           #Set Random State variable for randomizing splits on runs

# %%
#############################################################################
#
# Load Data
#
#####################
print('Loading Data...')
start_ts=time.time()

data = pd.read_csv("../../data/clean/model_plays.csv", low_memory = False)
data['target'].value_counts()


# Test for run and pass predictors. Comment out if we don't want this
data = data[((data['target'] == 'run') |  (data['target'] == 'pass'))]

# Separate Target from dataset
#firstdata['target'] = data['target'].astype('category')
#target_cat_label = dict(enumerate(data.target.categories))

# Change categorical variables to numerical
data['target'] = data['target'].astype('category').cat.codes
# data['target'] = data['target'].map(target_cat_label)

target = data['target']
del data['target']

# Split prior to any normalization/sampling. Stratify is set understanding that we have imbalanced classes and want to preserve that in the split
sample_train, sample_test, sample_target_train, sample_target_test = train_test_split(data, target, test_size=0.20, stratify = target)

data_train, data_test, target_train, target_test = train_test_split(sample_test, sample_target_test, test_size=0.20, stratify = sample_target_test)


# Calculates Shannon Entropy - a measure of class imbalance 
# As suggested by Simone Romano, PhD. 
# https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def balance(seq):
    n = len(seq)
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    k = len(classes)

    H = -sum([ (count/n) * np.log((count/n)) for clas,count in classes]) #shannon entropy
    return H/np.log(k)

print('The balance of classes within the dataset is: ', balance(target_train))
print('If this is far less than 1.0, please consider turning on the imblearn class balancing portions of the pipeline!')

print()    
print("Complete. Data Load Runtime:", time.time()-start_ts)
print()

# %%
#############################################################################
#
# Create pipeline lists 
#
##########################################


print('Creating Model Pipelines...')
start_ts=time.time()

# sklearn
#knn_pipe = []
#dt_pipe = []
#rf_pipe = []
#ab_pipe = []
#gb_pipe = []
#svm_pipe = []
#mlp_pipe = []
xg_pipe = []

#skpipes = [dt_pipe, rf_pipe, ab_pipe, gb_pipe, mlp_pipe, xg_pipe]
skpipes = [xg_pipe]

# %%
#############################################################################
#
# Pre-processing
#
##########################################

print('Normalization Turned On')

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
    print('Balanced Classes Turned On')
    for i in range(0, len(skpipes)):
        skpipes[i].append(('smote'+str(i), SMOTE(random_state = rand_st)))
        
elif imb_class == 2:
    # Undersample using NearMiss
    print('Balanced Classes Turned On')
    for i in range(0, len(skpipes)):
        skpipes[i].append(('NearMiss'+str(i), NearMiss(version=3)))
    
elif imb_class == 3:
    # Undersample using NearMiss
    print('Balanced Classes Turned On')
    for i in range(0, len(skpipes)):
        skpipes[i].append(('undersample'+str(i), RandomUnderSampler()))
    

# %%
#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('Low Variance Filter Turned On', '\n')
    
    for i in range(0, len(skpipes)):
        skpipes[i].append(('variance_threshold'+str(i), VarianceThreshold(threshold = 0.5)))   


#Feature Selection
if feat_select==1:
    print('Feature Selection Turned On', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None, class_weight="balanced")
        cv_rfc = RFECV(clf, n_features_to_select=k_cnt, step=.1, cv = 5, scoring = 'roc_auc')
            
        for i in range(0, len(skpipes)):
            skpipes[i].append(('rfe_rf'+str(i), cv_rfc))   
        
    if fs_type==2:
        #Wrapper Select via model
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
        sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
        
        for i in range(0, len(skpipes)):
            skpipes[i].append(('wrapper_rf'+str(i), sel))   

    if fs_type==3:                                                                ######Only work if the Target is binned###########
        #Univariate Feature Selection - Chi-squared
        #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
        print ('Univariate Feature Selection - Chi2: ')
        sel=SelectKBest(chi2, k=k_cnt)
        
        for i in range(0, len(skpipes)):
            skpipes[i].append(('ufs'+str(i), sel)) 


# %%
#############################################################################
#
# Dimensionality Reduction
#
##########################################

# PCA Dimensionality Reduction
if pca == 1:
    print('PCA Turned On', '\n')
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
classifiers = [#('knn_classifier', KNeighborsClassifier()),
               
               # Tree-based methods
               #('dt_classifier', DecisionTreeClassifier()), 
               #('rf_classifier', RandomForestClassifier()),
               #('ab_classifier', AdaBoostClassifier()),
               #('gb_classifier', GradientBoostingClassifier()),
               
               # Support Vector Machines
               #('sv_classifier', SVC()),
               
               # Neural Networks
               #('nn_classifier', MLPClassifier()),
               ('xg_classifier', xgb.XGBClassifier(tree_method = 'gpu_hist'))
               #('xg_classifier', xgb.XGBClassifier())
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
#knn_pipe = Pipeline(skpipes[0])


# Tree Methods
#dt_pipe = Pipeline(skpipes[0])
#rf_pipe = Pipeline(skpipes[1])
#ab_pipe = Pipeline(skpipes[2])
#gb_pipe = Pipeline(skpipes[3])
xg_pipe = Pipeline(skpipes[0])

# SVMs
#svm_pipe = Pipeline(skpipes[5])

# Neural Networks
#mlp_pipe = Pipeline(skpipes[4])




# List of all classifier pipelines
#pipelines = [dt_pipe, rf_pipe, ab_pipe, gb_pipe, mlp_pipe]
pipelines = [xg_pipe]
best_accuracy=0.0
best_classifier=0
best_pipeline=""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {1: 'Decision Tree', 2: 'RandomForest', 3: 'AdaBoost', 4:'GradientBoostedTrees', 5:'svm', 6:'feedforwardnn'}
#pipe_list = ['Decision Tree', 'RandomForest', 'AdaBoost', 'GradientBoostedTrees', 'feedforwardnn']
pipe_list = ['xgboost']


print("Complete. Pipeline Runtime:", time.time()-start_ts)
print()


# %% 
#############################################################################
#
# Hyperparameter Tuning
#
##########################################

best_params = []
best_estimator =[]

  
dt_param = [{'dt_classifier__criterion': ['gini', 'entropy'],
             'dt_classifier__splitter': ['best', 'random'],
             'dt_classifier__max_features': ['auto', 'sqrt', 'log2', None],
             'dt_classifier__max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
             'dt_classifier__min_samples_split': [2, 5, 10, 20, 25, 50, 100],
             'dt_classifier__min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 128, 256],
             'dt_classifier__class_weight': ['balanced', 'balanced_subsample', None]}]

rf_grid = [{'rf_classifier__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
            'rf_classifier__max_features': ['auto', 'sqrt', 'log2'],
            'rf_classifier__max_depth': [int(x) for x in np.linspace(1, 20, num = 20)],
            'rf_classifier__min_samples_split': [2, 5, 10, 20, 25, 50, 100, 200],
            'rf_classifier__min_samples_leaf': [2, 4, 8, 16, 32, 64, 128],
            'rf_classifier__class_weight':  ['balanced', 'balanced_subsample', None],
            'rf_classifier__bootstrap': [True, False]}]

ab_grid = [{'ab_classifier__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
           'ab_classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
           'ab_classifier__algorithm': ['SAMME', 'SAMME.R']}]


gb_grid = [{'gb_classifier__loss': ['deviance', 'exponential'],
           'gb_classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
           'gb_classifier__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
           'gb_classifier__max_depth' : [3,4,5,6,7,8,9,10,11,12,13,14,15],
           'gb_classifier__max_features' : ['auto','sqrt','log2',None],
           'gb_classifier__min_samples_split': [2, 5, 10, 20, 25, 50, 100],
           'gb_classifier__min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
           }]


nn_grid = [{'nn_classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
           'nn_classifier__hidden_layer_sizes' : [(100,), (10,), (50,50), (20,20), (100,50), (200,100), (30,30,30)],
           'nn_classifier__solver': ['lbfgs', 'sgd', 'adam'],
           'nn_classifier__alpha' : [0.0001, 0.001, 0.01],
           'nn_classifier__learning_rate' : ['constant', 'invscaling', 'adaptive'],
           'nn_classifier__max_depth' : [3,4,5,6,7,8],
           'nn_classifier__max_features' : ['auto','sqrt','log2',None],
           'nn_classifier__min_samples_split': [2, 5, 10, 20, 25, 50, 100],
           'nn_classifier__min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
           }]


xg_grid = [{'xg_classifier__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
            #'xg_classifier__learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.150, 0.25, 0.5, 1],
            #'xg_classifier__min_split_loss': [0, 0.5, 1, 1.5, 2],
            #'xg_classifier__min_child_weight': [0, 1, 3, 5, 7],
            #'xg_classifier__max_depth': [3, 5, 7, 9, 11, 13,15, 17, 19, 21, 25],
            'xg_classifier__max_depth': [3, 5, 7, 9],
            'xg_classifier__booster': ['gbtree', 'gblinear', 'dart'],
            #'xg_classifier__reg_alpha': [0, 0.001, 0.005, 0.1, 0.05, 1e-5, 0.1, 0.25, 0.5, 1, 1.25],
            #'xg_classifier__reg_lambda': [0, 1, 1.1, 1.2, 1.3],
            #'xg_classifier__sampling_method' : ['gradient_based', 'uniform'],
            #'xg_classifier__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #'xg_classifier__colsample_bytree' : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
           }]


#params = [dt_param, rf_grid, ab_grid, gb_grid, nn_grid]
params = [xg_grid]

if param_tuning == 1:
    
    #scorers = {'AUC' : 'roc_auc'}
    
    for pipe, grid_param, name in zip(pipelines, params, pipe_list):
        print('Tuning Models...')
        start_ts=time.time()
        if exhaustive_search == 1:
            gridsearch = GridSearchCV(pipe, xg_grid, scoring = 'roc_auc', cv=5, verbose=10, n_jobs=-1) 
        else:    
            gridsearch = RandomizedSearchCV(pipe, param_distributions = grid_param, n_iter=1000, scoring = 'roc_auc', cv=5, verbose=10, n_jobs=-1) 
        gridsearch.fit(data_train, target_train)
 
        # Get best results        
        best_params.append((name, gridsearch.best_params_, gridsearch.best_score_))
        print(gridsearch.best_params_, file=open('output.txt', 'a'))
        print(gridsearch.best_score_, file=open('output.txt', 'a'))
        
        # Save optimized model
        filename = name + '.sav'
        best_model = gridsearch.best_estimator_
        pickle.dump(best_model, open(filename, 'wb'))
        best_estimator.append(gridsearch.best_estimator_)
        
        print("Complete. GridSearchCV Runtime:", time.time()-start_ts)
        print()
        
        
#############################################################################
#
# Train SciKit Models
#
##########################################

print('--ML Model Output--', '\n')

# Non-Cross Validation
if cross_val == 0:
    
    feature_importances = []
    
    # Decision Tree
    clf = DecisionTreeClassifier(criterion='gini', 
                                 splitter='best', 
                                 max_depth=10, 
                                 min_samples_split=2, 
                                 min_samples_leaf=32, 
                                 max_features=None,
                                 class_weight='balanced',
                                 random_state=rand_st)
    clf.fit(data_train, target_train)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("Decision Tree Train Accuracy:",clf.score(data_train, target_train))
    print('Decision Tree Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    feature_importances.append(('decision tree', clf.feature_importances_))
    
    # Random Forest
    clf = RandomForestClassifier(n_estimators=1000, 
                                 max_depth=None, 
                                 min_samples_split=5, 
                                 min_samples_leaf = 2, 
                                 max_features = 'auto',                                 
                                 random_state=rand_st)
    clf.fit(data_train, target_train)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("Random Forest Train Accuracy:",clf.score(data_train, target_train))
    print('Random Forest Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    feature_importances.append(('random forest', clf.feature_importances_))

    # AdaBoost
    clf=AdaBoostClassifier(n_estimators = 950,
                           base_estimator = None,
                           learning_rate = 0.1,
                           algorithm = 'SAMME.R',
                           random_state = rand_st)
    clf.fit(data_train, target_train)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("AdaBoost Train Accuracy:",clf.score(data_train, target_train))
    print('AdaBoost Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    feature_importances.append(('adaboost', clf.feature_importances_))

    # Gradient Boosting
    clf=GradientBoostingClassifier(n_estimators = 850, 
                                   loss = 'deviance', 
                                   learning_rate = 0.01, 
                                   max_depth = 7, 
                                   min_samples_split = 20, 
                                   min_samples_leaf = 2,
                                   random_state = rand_st)
    clf.fit(data_train, target_train)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("Gradient Boosting Train Accuracy:",clf.score(data_train, target_train))
    print('Gradient Boosting Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    feature_importances.append(('gradient boost', clf.feature_importances_))
    
     # Neural Network
    clfnn=MLPClassifier(activation = 'logistic',
                      learning_rate = 'adaptive',
                      solver = 'adam',
                      alpha = 0.01,
                      hidden_layer_sizes = (200,100), 
                      random_state = rand_st)
    clfnn.fit(data_train, target_train)
    test_predictions = clfnn.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("NN Train Accuracy:",clf.score(data_train, target_train))
    print('NN Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    
    # Catboost
    clf=CatBoostClassifier(task_type = 'GPU', silent = True)
    clf.fit(data_train, target_train)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("CatBoost Train Accuracy:",clf.score(data_train, target_train))
    print('CatBoost Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))       
    feature_importances.append(('Catboost', clf.feature_importances_))
    
    # XGBoost
    clf=xgb.XGBClassifier()
    clf.fit(data_train, target_train)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("XGBoost Train Accuracy:",clf.score(data_train, target_train))
    print('XGBoost Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    feature_importances.append(('XGboost', clf.feature_importances_))

    clf.save_model('../../models/xgbmodel.bst')

    
    fi = []
    # Built-in Feature Importances
    for f in feature_importances:
        feat = pd.DataFrame(data_train.columns)
        values = pd.DataFrame(f[1])
        fi.append((f[0],feat.join(values, lsuffix = 'val')))
        
        
