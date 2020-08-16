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
from sklearn.pipeline import make_pipeline, Pipeline
#from imblearn.pipeline import make_pipeline, Pipeline

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
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
import xgboost.sklearn as xgb
from catboost import CatBoostClassifier, cv, Pool
from lightgbm.sklearn import LGBMClassifier
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers

# Model Evaluation
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer

# Model Exporting
#from sklearn.externals import joblib
import pickle

# Model Interpretability
import shap

from numpy.ma import MaskedArray
import sklearn.utils.fixes
import matplotlib.pyplot as plt

sklearn.utils.fixes.MaskedArray = MaskedArray

# pip install tune-sklearn ray[tune]
#from tune_sklearn import TuneSearchCV
#from tune_sklearn import TuneGridSearchCV
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# ML Flow
#import mlflow.sklearn


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
cross_val= 0                                        #Control Switch for CV
base_classifiers = 0                                #Apply base classifier as baseline if CV is turned on
norm_features=1                                     #Normalize features switch
pca = 0
feat_select=0                                       #Control Switch for Feature Selection
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         #Control switch for low variance filter on features
k_cnt= 20                                           #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
param_tuning = 0                                    #Turn on model parameter tuning
exhaustive_search = 0                               #Turn on if you want exhaustive grid search. Otherwise, it will default to RandomizedSearchCV
neural_network = 0                                  #Turn on if you want to train a neural network. Does not use pipelineing structure!!!
feature_importance = 1                              #Turn on shap if cross_val = 0

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




# Result of Feature Selection - Other Script
data.drop(['play_type_punt', 'Fog', 'ld_outcome_end_of_half', 'pd_expl_pass', 'play_type_field_goal', 'play_type_qb_kneel', 'GameMonth', 'Wind', 
           'ld_opp_outcome_end_of_half', 'ld_opp_outcome_punt', 'ld_outcome_field_goal', 'pd_average_tfl', 'Rain', 'Snow', 'def_fs', 'def_le',
           'def_re', 'def_rolb', 'def_ss', 'fumble', 'home', 'ld_outcome_fumble_lost', 'ld_outcome_interception', 'ld_outcome_touchdown',
           'off_lg', 'off_rg', 'off_te', 'def_cb', 'def_dt', 'def_mlb', 'ld_drive_length', 'ld_expl_pass', 'ld_plays', 'off_rt', 'pd_pass_yard_att',
           'qb_scramble', 'def_lolb', 'ld_opp_outcome_field_goal', 'ld_opp_outcome_interception', 'ld_opp_outcome_no_ld', 'ld_outcome_no_ld',
           'ld_outcome_punt', 'ld_outcome_turnover_on_downs', 'off_c', 'qb_spike', 'play_type_qb_spike' , 'pd_average_plays', 'pd_average_sacks', 'pd_average_top'
           , 'pd_avg_interceptions'], axis = 1, inplace = True)


# Test for run and pass predictors. Comment out if we don't want this
data = data[((data['target'] == 'run') |  (data['target'] == 'pass'))]

# Change categorical variables to numerical
data['target'] = data['target'].astype('category').cat.codes
# data['target'] = data['target'].map(target_cat_label)

# Create 2019 Bears Dataset
bears = data.copy()
bears = bears[((bears['posteam'] == 'CHI') & (bears['GameYear'] == 2019))]

del data['posteam']
del data['GameYear']
del bears['posteam']
del bears['GameYear']
bears_target = bears['target']
del bears['target']


target = data['target']
del data['target']


# Separate Target from dataset
#firstdata['target'] = data['target'].astype('category')
#target_cat_label = dict(enumerate(data.target.categories))


# Split prior to any normalization/sampling. Stratify is set understanding that we have imbalanced classes and want to preserve that in the split

# If doing model evaluation use entire dataset
if cross_val == 1:
    data_train = data.copy()
    target_train = target.copy()
else:
    #If hyperparameter tuning, use subset of data for speed purposes
    sample_train, sample_test, sample_target_train, sample_target_test = train_test_split(data, target, test_size=0.50, stratify = target)
    data_train, data_test, target_train, target_test = train_test_split(sample_train, sample_target_train, test_size=0.50, stratify = sample_target_train)
    data_shap, data_shap_test, target_shap, target_shap_test = train_test_split(data_test, target_test, test_size=0.98, stratify = target_test)

# Remove labels
data_ids = data[['play_id', 'game_id']]
del data['play_id']
del data['game_id']

data_train_ids = data_train[['play_id', 'game_id']]
del data_train['play_id']
del data_train['game_id']

data_test_ids = data_test[['play_id', 'game_id']]
del data_test['play_id']
del data_test['game_id']

data_shap_ids = data_shap[['play_id', 'game_id']]
del data_shap['play_id']
del data_shap['game_id']


bears_copy = bears.copy()
bears_copy.to_csv('bears2019.csv', index = False)
bears_target.to_csv('bears2019target.csv', index = False)
bears_ids = bears[['play_id', 'game_id']]
del bears['play_id']
del bears['game_id']



# Split la


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
knn_pipe = []
log_pipe = []
dt_pipe = []
rf_pipe = []
ab_pipe = []
gb_pipe = []
xg_pipe = []
lgbm_pipe = []
cat_pipe = []
svm_pipe = []
svc_pipe = []

skpipes = [knn_pipe, log_pipe, dt_pipe, rf_pipe, ab_pipe, gb_pipe, xg_pipe, lgbm_pipe, cat_pipe, svm_pipe, svc_pipe]

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
# NEURAL NETWORK TRAINING USING KERAS
#
##########################################

scaler = StandardScaler()
data_np = pd.DataFrame(scaler.fit_transform(data_train), columns = data_train.columns)


# I understand that this is very inelegant. For now it will do
if neural_network == 1:
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=512, activation='relu', input_shape=(len(data_train.columns),)))

    network.add(layers.Dense(units=512, activation='relu'))

    network.add(layers.Dropout(0.4))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=256, activation='relu'))
    
    network.add(layers.Dense(units=256, activation='relu'))
    
    network.add(layers.Dropout(0.4))
    
    network.add(layers.Dense(units=128, activation='relu'))
    
    network.add(layers.Dense(units=128, activation='relu'))
    
    network.add(layers.Dropout(0.4))
    
    network.add(layers.Dense(units=64, activation='relu'))
    
    network.add(layers.Dense(units=64, activation='relu'))
    
    network.add(layers.Dropout(0.4))
    
    network.add(layers.Dense(units=32, activation='relu'))
    
    network.add(layers.Dense(units=32, activation='relu'))
    
    network.add(layers.Dropout(0.4))
    
    network.add(layers.Dense(units=16, activation='relu'))
    
    network.add(layers.Dense(units=16, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))
    
    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='Adam', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    network.summary()
    
    # Train neural network
    history = network.fit(data_np, # Features
                          target_train, # Target vector
                          epochs=50, # Number of epochs
                          verbose=1, # Print description after each epoch
                          batch_size=100, # Number of observations per batch
                          validation_data=(data_test, target_test)) # Data for evaluation
    
    # Plot accuracy for training and validation datasets
    plt.plot(history.history['accuracy'], label='train_accuracy') # For TF2
    plt.plot(history.history['val_accuracy'], label = 'valid_accuracy') # For TF2
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    # Evaluate the learned model with validation set
    valid_loss, valid_acc = network.evaluate(data_test, target_test, verbose=2) #
    print ("valid_accuracy=%s, valid_loss=%s" % (valid_acc, valid_loss))
    

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
classifiers = [('knn_classifier', KNeighborsClassifier(n_jobs = -1)),
               ('log_classifier', LogisticRegression(max_iter = 1000)),
               
               # Tree-based methods
               ('dt_classifier', DecisionTreeClassifier()), 
               ('rf_classifier', RandomForestClassifier()),
               ('ab_classifier', AdaBoostClassifier()),
               ('gb_classifier', GradientBoostingClassifier()),
               ('xg_classifier', xgb.XGBClassifier()),
               ('lgbm_classifier', LGBMClassifier()),
               ('cat_classifier', CatBoostClassifier()),
               
               # Support Vector Machines
               ('sv_classifier', LinearSVC()),
               ('svc_classifier', SVC(cache_size = 1000, max_iter = 5000))

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

# Other Classifiers
knn_pipe = Pipeline(skpipes[0])
log_pipe = Pipeline(skpipes[1])


# Tree Methods
dt_pipe = Pipeline(skpipes[2])
rf_pipe = Pipeline(skpipes[3])
ab_pipe = Pipeline(skpipes[4])
gb_pipe  = Pipeline(skpipes[5])
xg_pipe = Pipeline(skpipes[6])
lgbm_pipe = Pipeline(skpipes[7])
cat_pipe = Pipeline(skpipes[8])

# SVMs
svm_pipe = Pipeline(skpipes[9])
svc_pipe = Pipeline(skpipes[10])


# List of all classifier pipelines
pipelines = [cat_pipe]
#pipelines = [knn_pipe, log_pipe, dt_pipe, rf_pipe, ab_pipe, gb_pipe, xg_pipe, lgbm_pipe, cat_pipe, svm_pipe, svc_pipe]

best_accuracy=0.0
best_classifier=0
best_pipeline=""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {1: 'Decision Tree', 2: 'RandomForest', 3: 'AdaBoost', 4:'GradientBoostedTrees', 5:'svm', 6:'feedforwardnn'}
#pipe_list = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'AdaBoost', 'GradientBoostedTrees', 'XGBoost', 'LightGBM', 'CatBoost', 'LinearSVC', 'SVC']
pipe_list = ['CatBoost']

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

  
knn_grid = [{'knn_classifier__n_neighbors' : [3,5,7,9,11],
              'knn_classifier__weights' : ['uniform', 'distance'],
              'knn_classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'knn_classifier__p' : [1,2]}]


log_grid = {'log_classifier__solver': Categorical(['saga']),  
            'log_classifier__penalty': Categorical(['l1','l2', 'elasticnet', 'none']),
            'log_classifier__tol': Real(1e-5, 1e-3, 'log-uniform'),
            'log_classifier__C': Real(1e-5, 100, 'log-uniform'),
            'log_classifier__class_weight': Categorical(['balanced', None]),
            'log_classifier__fit_intercept': Categorical([True, False]),
            'log_classifier__l1_ratio': Real(0,1)
            }

''' LBFGS Scenario
log_grid = {'log_classifier__solver': Categorical(['lbfgs', 'newton-cg','sag']),  
            'log_classifier__penalty': Categorical(['l2', 'none']),
            'log_classifier__tol': Real(1e-5, 1e-3, 'log-uniform'),
            'log_classifier__C': Real(1e-5, 100, 'log-uniform'),
            'log_classifier__class_weight': Categorical(['balanced', None]),
            'log_classifier__fit_intercept': Categorical([True, False])
            }
'''

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

'''
gb_grid = [{'gb_classifier__loss': ['deviance', 'exponential'],
           'gb_classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
           'gb_classifier__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
           'gb_classifier__max_depth' : [3,4,5,6,7,8,9,10,11,12,13,14,15],
           'gb_classifier__max_features' : ['auto','sqrt','log2',None],
           'gb_classifier__min_samples_split': [2, 5, 10, 20, 25, 50, 100],
           'gb_classifier__min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
           }]
'''

gb_grid = {'gb_classifier__loss': ['deviance', 'exponential'],
           'gb_classifier__learning_rate': (0.0001, 1),
           'gb_classifier__n_estimators': (0, 1000),
           'gb_classifier__max_depth' : (3,15),
           'gb_classifier__max_features' : ['auto','sqrt','log2',None],
           'gb_classifier__min_samples_split': (2, 100),
           'gb_classifier__min_samples_leaf': (1, 128)
           }


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


xg_grid = [{#'xg_classifier__n_estimators': [int(x) for x in np.linspace(start = 1000, stop = 1500, num = 6)],
            'xg_classifier__n_estimators': [1400],
            #'xg_classifier__learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.150, 0.25, 0.5, 1],
            'xg_classifier__learning_rate': [0.01],
            'xg_classifier__min_split_loss': [2],
            #'xg_classifier__min_child_weight': [0, 1, 3, 5, 7],
            'xg_classifier__min_child_weight': [7],
            #'xg_classifier__gamma': [i/10.0 for i in range(0,5)],
            'xg_classifier__gamma': [0],
            #'xg_classifier__max_depth': [3, 5, 7, 9, 11, 13,15, 17, 19, 21, 25],
            'xg_classifier__max_depth': [7],
            #'xg_classifier__booster': ['gbtree', 'gblinear', 'dart'],
            'xg_classifier__booster': ['gbtree'],
            #'xg_classifier__reg_alpha': [0, 0.001, 0.005, 0.1, 0.05, 1e-5, 0.1, 0.25, 0.5, 1, 1.25],
            'xg_classifier__reg_alpha': [0.05],
            'xg_classifier__reg_lambda': [0, 1, 1.1, 1.2, 1.3],
            #'xg_classifier__sampling_method' : ['gradient_based', 'uniform'],
            'xg_classifier__subsample': [0.9],
            #'xg_classifier__subsample': [0.8],
            'xg_classifier__colsample_bytree' : [ 0.6]
            #'xg_classifier__colsample_bytree' : [0.8]
            #'xg_classifier__early_stopping_rounds' : [50],
            #'xg_classifier__eval_metric' : ['auc'],
            #'xg_classifier__eval_set' : [[data_test, target_test]]
           }]


# Using TuneSearchCV
lgbm_grid ={'lgbm_classifier__n_iterations': Integer(1, 1500),
            #'lgbm_classifier__boosting_type' : Categorical(['gbdt', 'rf', 'goss', 'dart']),
            'lgbm_classifier__boosting_type' : Categorical(['gbdt']),
            'lgbm_classifier__learning_rate': Real(0.000001, 1),
            'lgbm_classifier__class_weight' : Categorical(['balanced', None]),
            'lgbm_classifier__num_leaves' : Integer(2, 50),
            'lgbm_classifier__tree_learner': Categorical(['serial', 'feature', 'data', 'voting']),
            'lgbm_classifier__colsample_bytree' : Real(.1, 1.0),
            'lgbm_classifier__subsample' : Real(0.0001, 0.9),
            'lgbm_classifier__subsample_freq' : Integer(1, 100),
            'lgbm_classifier__min_split_gain' : Real(0, 0.9),
            'lgbm_classifier__reg_alpha' : Real(0, 1),
            'lgbm_classifier__reg_lambda' : Real(0, 1)
           }


cat_grid = {'cat_classifier__depth' : [2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16],
            'cat_classifier__l2_leaf_reg' : [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10],
            'cat_classifier__learning_rate' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
            'cat_classifier__bagging_temperature' : [0, 0.5, 1, 2, 5, 10],
            'cat_classifier__grow_policy' : ['SymmetricTree', 'Depthwise'],
            'cat_classifier__border_count' : [1,5, 10, 25, 50, 75, 100, 125, 255],
            'cat_classifier__iterations' : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'cat_classifier__loss_function' : ['Logloss', 'CrossEntropy']
            #'cat_classifier__max_leaves' : Integer(1,64)
            }

'''
cat_grid = {'cat_classifier__depth' : Integer(1,16),
            'cat_classifier__l2_leaf_reg' : Real(0.001, 10),
            'cat_classifier__learning_rate' : Real(0.0001, 1),
            'cat_classifier__bagging_temperature' : Real(0, 10),
            'cat_classifier__grow_policy' : Categorical(['SymmetricTree', 'Depthwise']),
            'cat_classifier__border_count' : Integer(1,255),
            'cat_classifier__iterations' : Integer(100,1000),
            'cat_classifier__loss_function' : Categorical(['Logloss', 'CrossEntropy'])
            #'cat_classifier__max_leaves' : Integer(1,64)
            }
'''

sv_grid = [{'sv_classifier__penalty' : ['l2'],
           'sv_classifier__loss' : ['squared_hinge'],
           'sv_classifier__dual' : [True],
           'sv_classifier__class_weight' : ['balanced', None],
           'sv_classifier__C' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
           'sv_classifier__max_iter' : [5000]
            }]

svc_grid = [{'svc_classifier__kernel' : Categorical(['poly', 'rbf', 'sigmoid']),
           'svc_classifier__degree' : Integer(2,7),
           'svc_classifier__gamma' : Categorical(['scale', 'auto']),
           'svc_classifier__class_weight' : Categorical(['balanced', None]),
           'svc_classifier__tol' : Real(1e-5, 1e-2),
           'svc_classifier__C' : Real(0.1, 0.9),
           'svc_classifier__coef0' : Real(0.1, 0.9),
           'svc_classifier__shrinking': Categorical([True, False])
            }]



############################
# Hard Coded Optimal Params
############################

knn_opt_grid = {'knn_classifier__n_neighbors' : 11,
                 'knn_classifier__weights' : 'uniform',
                 'knn_classifier__algorithm': 'auto',
                 'knn_classifier__p' : 1}

log_opt_grid = {'log_classifier__solver': 'saga',  
                'log_classifier__penalty': 'l1',
                'log_classifier__tol': 4.624212298955742e-05,
                'log_classifier__C': 1.3122026146734938,
                'log_classifier__class_weight': 'balanced',
                'log_classifier__fit_intercept': True
                }

dt_opt_grid = {'dt_classifier__criterion': 'gini',
               'dt_classifier__splitter': 'best',
               'dt_classifier__max_features': None,
               'dt_classifier__max_depth': 9,
               'dt_classifier__min_samples_split': 2,
               'dt_classifier__min_samples_leaf': 128,
               'dt_classifier__class_weight': None}

rf_opt_grid = {'rf_classifier__n_estimators': 900,
               'rf_classifier__max_features': 'sqrt',
               'rf_classifier__max_depth': 17,
               'rf_classifier__min_samples_split': 20,
               'rf_classifier__min_samples_leaf': 4,
               'rf_classifier__class_weight':  None,
               'rf_classifier__bootstrap': False}

ab_opt_grid = {'ab_classifier__n_estimators': 800,
               'ab_classifier__learning_rate': 0.5,
               'ab_classifier__algorithm': 'SAMME.R'}

gb_opt_grid = {'gb_classifier__loss': 'deviance',
               'gb_classifier__learning_rate': 0.3958765328906512,
               'gb_classifier__n_estimators': 862,
               'gb_classifier__max_depth' : 7,
               'gb_classifier__max_features' : None,
               'gb_classifier__min_samples_split': 66,
               'gb_classifier__min_samples_leaf': 105
               }

xg_opt_grid = {'xg_classifier__n_estimators': 1400,
               'xg_classifier__learning_rate': 0.01,
               'xg_classifier__min_split_loss': 2,
               'xg_classifier__min_child_weight': 7,
               'xg_classifier__gamma': 0,
               'xg_classifier__max_depth': 7,
               'xg_classifier__booster': 'gbtree',
               'xg_classifier__reg_alpha': 0.05,
               'xg_classifier__reg_lambda': 0,
               'xg_classifier__subsample': 0.9,
               'xg_classifier__colsample_bytree' : 0.6
               }

lgbm_opt_grid ={#'lgbm_classifier__n_iterations': 1,
                'lgbm_classifier__n_iterations': 207,
                'lgbm_classifier__boosting_type' : 'gbdt',
                'lgbm_classifier__learning_rate': 0.06757615716871256,
                'lgbm_classifier__class_weight' : None,
                'lgbm_classifier__num_leaves' : 5,
                'lgbm_classifier__tree_learner': 'feature',
                'lgbm_classifier__colsample_bytree' : 0.8436227416617349,
                'lgbm_classifier__subsample' : 0.9,
                'lgbm_classifier__subsample_freq' : 1,
                'lgbm_classifier__min_split_gain' : 0.9,
                'lgbm_classifier__reg_alpha' : 1.0,
                'lgbm_classifier__reg_lambda' : 1.0
                }

cat_opt_grid = {'cat_classifier__depth' : 6,
                'cat_classifier__l2_leaf_reg' : 0.8412407155468501,
                'cat_classifier__bagging_temperature' : 0,
                'cat_classifier__grow_policy' : 'SymmetricTree'
                }

sv_opt_grid = {'sv_classifier__penalty' : 'l1',
               'sv_classifier__loss' : 'squared_hinge',
               'sv_classifier__dual' : False,
               'sv_classifier__class_weight' : 'balanced',
               'sv_classifier__C' : 0.1,
               'sv_classifier__max_iter' : 5000
               }

svc_opt_grid = {'svc_classifier__kernel' : 'rbf',
                'svc_classifier__degree' : 5,
                'svc_classifier__gamma' : 'auto',
                'svc_classifier__class_weight' : None,
                'svc_classifier__tol' : 1e-05,
                'svc_classifier__C' : 0.1,
                'svc_classifier__coef0' : 0.6976330580565031,
                'svc_classifier__shrinking': False
                }


params = [cat_grid]

opt_params = [knn_opt_grid, log_opt_grid, dt_opt_grid, rf_opt_grid, ab_opt_grid, gb_opt_grid, xg_opt_grid, lgbm_opt_grid, cat_opt_grid, sv_opt_grid, svc_opt_grid ]

if param_tuning == 1:
    
    gridhistory = []
    
    #scorers = {'AUC' : 'roc_auc'}
    
    for pipe, grid_param, name in zip(pipelines, params, pipe_list):
        print('Tuning Models...')
        start_ts=time.time()
        if exhaustive_search == 1:
            gridsearch = GridSearchCV(pipe, grid_param, scoring = 'roc_auc', cv=5, verbose=10, n_jobs=-1) 
            #gridsearch = TuneGridSearchCV(pipe, grid_param, scoring = 'roc_auc', cv=5, verbose=10, n_jobs=-1) 
        else:    
            gridsearch = RandomizedSearchCV(pipe, param_distributions = grid_param, n_iter=200, scoring = 'roc_auc', cv=5, verbose=20, n_jobs=-1)
            #gridsearch = TuneSearchCV(pipe, param_distributions = grid_param, n_iter = 2, cv = 2, scoring = 'roc_auc', search_optimization='bayesian', verbose=2, n_jobs = -1)
            #gridsearch = BayesSearchCV(pipe, search_spaces = grid_param, n_iter = 200, scoring = 'roc_auc', n_jobs = -1, verbose = 10)
        gridsearch.fit(data_train, target_train)
 
        # Get best results        
        best_params.append((name, gridsearch.best_params_, gridsearch.best_score_))
        print(gridsearch.best_params_, file=open('output.txt', 'a'))
        print(gridsearch.best_score_, file=open('output.txt', 'a'))
        
        # Save optimized model
        gridhistory.append(gridsearch.cv_results_)
        
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


# Non-Cross Validation - Train model save, and generate SHAP Values
if cross_val == 0:
    
    feature_importances = []
          
    # XGBoost
    clf = xg_pipe
    clf.set_params(**xg_opt_grid)
    clf.fit(data, target)
    
    pickle.dump(clf, open('xgb_model_final.sav', 'wb'))
    #clf.save_model('../../models/xgbmodel_final.bst')
    
    # Built-in Feature Importances
    feature_importances.append(('XGboost', clf.named_steps['xg_classifier'].feature_importances_))
    fi = []
    
    for f in feature_importances:
        feat = pd.DataFrame(data_train.columns)
        values = pd.DataFrame(f[1])
        fi.append((f[0],feat.join(values, lsuffix = 'val')))
    pd.DataFrame(fi[0][1]).to_csv('features.csv')    

    
    # Bears Predictions
    bears_predictions = clf.predict(bears)
    scores_ACC = clf.score(bears, bears_target)
    print("XGBoost Train Accuracy:",clf.score(data, target))
    print('XGBoost Test Acc:', scores_ACC)
    print(classification_report(bears_target, bears_predictions))
    bears_predictions = pd.DataFrame(bears_predictions, columns = ['Predicted'])
    bears_predictions.to_csv('Bears Predictions.csv')
    
    # SHAP Implementation    
    xg_model = clf.named_steps['xg_classifier']
    xg_model = xg_model.get_booster()
    standard_scaler = clf.named_steps['scalar6']    
    
    model_bytearray = xg_model.save_raw()[4:]
    def myfun(self=None):
        return model_bytearray
    xg_model.save_raw = myfun
    

    data_shap = standard_scaler.transform(data_shap)
    bears = standard_scaler.transform(bears)
    
    if feature_importance == 1:
        shap_explainer = shap.TreeExplainer(xg_model, data_shap)
        shap_values = shap_explainer.shap_values(bears)
        
        bears_ids = bears_ids.reset_index(drop=True)
        result = pd.concat([bears_ids, pd.DataFrame(shap_values, columns = data.columns)], axis = 1, sort=False)
        result.to_csv('bears_shap.csv', index = False)
                
        
####Cross-Val Classifiers####
if cross_val == 1:
    
    classifier_cross_val = pd.DataFrame(data = None, columns = ['Classifier' , 'Accuracy', 'Balanced Accuracy' , 'Precision', 'Average Precision', 'Recall', 'F1', 'AUC']) 
    
    #Setup Crossval classifier scorers
    scorers = {'Accuracy': 'accuracy', 
               'Balanced_Accuracy': 'balanced_accuracy',
               'Precision': 'precision',
               'Average_Precision': 'average_precision',
               'Recall': 'recall',
               'F1': 'f1',
               'AUC' : 'roc_auc'}

    for pipe, param, name in zip(pipelines, opt_params, pipe_list):
        #SciKit Logistic Regression  - Cross Val
        start_ts=time.time()
        if base_classifiers == 0:
            #pipe.set_params(**param)
            cat_pipe.set_params(**cat_opt_grid)
        clf = cat_pipe
        scores = cross_validate(clf, data_train, target_train, scoring=scorers, cv= 10)
        scores_Acc = scores['test_Accuracy']
        scores_Bal = scores['test_Balanced_Accuracy']
        scores_Pre = scores['test_Precision']
        scores_avPre = scores['test_Average_Precision']
        scores_Rec = scores['test_Recall']
        scores_F1 = scores['test_F1']
        scores_AUC = scores['test_AUC']
        classifier_cross_val = classifier_cross_val.append({'Classifier': name, 'Accuracy' : "%0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2), 
                                                            'Balanced Accuracy' : "%0.4f (+/- %0.4f)" % (scores_Bal.mean(), scores_Bal.std() * 2),
                                                            'Precision' : "%0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2),
                                                            'Average Precision' : "%0.4f (+/- %0.4f)" % (scores_avPre.mean(), scores_avPre.std() * 2),
                                                            'Recall' : "%0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2),
                                                            'F1' : "%0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2),
                                                            'AUC' : "%0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2)
                                                            },
                                                           ignore_index = True)
        print(name, "CV Runtime:", time.time()-start_ts)
        
    if base_classifiers == 1:
        classifier_cross_val.to_csv('baseline_classifier_scores.csv')
    else:
        classifier_cross_val.to_csv('tuned_classifier_scores.csv')
