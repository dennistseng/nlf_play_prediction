#Scikit Template
'''created by Casey Bennett 2018, www.CaseyBennett.com
   Copyright 2018

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License (Lesser GPL) as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   Modified by Dennis Tseng for DSC 672 Capstone Project for Feature Engineering usage
'''

import sys
import csv
import math
import numpy as np
import pandas as pd
from operator import itemgetter
import time

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier, cv, Pool

#from sklearn.externals import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, scale, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

import shap
import lime

from collections import Counter

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


#############################################################################
#
# Global parameters
#
#####################

feat_select=1                                       #Control Switch for Feature Selection
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         #Control switch for low variance filter on features
k_cnt= 50                                           #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
param_tuning = 0                                    #Turn on model parameter tuning
feat_start = 0


#Set global model parameters
rand_st=0                                           #Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#####################

data = pd.read_csv("../../data/clean/model_plays.csv", low_memory = False)

# Test for run and pass predictors. Comment out if we don't want this
data = data[((data['target'] == 'run') |  (data['target'] == 'pass'))]

# Remove labels
del data['play_id']
del data['game_id']

# Separate Target from dataset
#firstdata['target'] = data['target'].astype('category')
#target_cat_label = dict(enumerate(data.target.categories))

# Change categorical variables to numerical
data['target'] = data['target'].astype('category').cat.codes
# data['target'] = data['target'].map(target_cat_label)

target = data['target']
del data['target']


# Split prior to any normalization/sampling. Stratify is set understanding that we have imbalanced classes and want to preserve that in the split
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20, stratify = target)


#############################################################################
#
# Classifier Parameter Tuning
#
##########################################

# Normalize features
scaler = StandardScaler()
data_np = pd.DataFrame(scaler.fit_transform(data_train), columns = data_train.columns)

# For reference
if param_tuning == 1:
    rf_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
               'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth': [int(x) for x in np.linspace(1, 10, num = 10)],
               'min_samples_split': [2, 5, 10, 20, 25, 50, 100],
               'min_samples_leaf': [2, 4, 8, 16, 32, 64],
               'class_weight': ['balanced', 'balanced_subsample', None],
               'bootstrap': [True, False]}
    
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 100, scoring= 'roc_auc', cv = 3, verbose=10, n_jobs = 4)
    
    # Fit the random search model
    rf_random.fit(data_np, target_train)
    
    print(rf_random.best_params_, file=open('feature_engineering.txt', 'a'))
    print(rf_random.best_score_, file=open('feature_engineering.txt', 'a'))
    print(rf_random.best_params_)
    print(rf_random.best_score_)


#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('--LOW VARIANCE FILTER ON--', '\n')
    
    #LV Threshold
    sel = VarianceThreshold(threshold=0.50)                                      #Removes any feature with less than 20% variance
    fit_mod=sel.fit(data_np)
    fitted=sel.transform(data_np)
    sel_idx=fit_mod.get_support()
    variance_features = list(data_np.columns[np.array(sel_idx).astype(bool)])
    
    print('Selected', variance_features)
    print('Features (total, selected):', len(data_np.columns), len(variance_features))
    print('\n')

    #Filter selected columns from original dataset
    data_np = data_np[variance_features]


#Feature Selection
if feat_select==1:
    '''
    Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
    ''' 
    
    print('--FEATURE SELECTION ON--', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        clf = RandomForestClassifier(n_estimators=400, 
                             max_depth=10, 
                             min_samples_split=20, 
                             min_samples_leaf = 8, 
                             max_features = 'auto',                                 
                             class_weight=  None,
                             bootstrap= True,
                             random_state=rand_st)
        sel = RFE(clf, n_features_to_select=k_cnt, step=.1)
        print('Stepwise Recursive Backwards - Random Forest: ')
            
        fit_mod=sel.fit(data_np, target_train)
        print(sel.ranking_)
        sel_idx=fit_mod.get_support()      

    if fs_type==2:
        #Wrapper Select via model
        clf = RandomForestClassifier(n_estimators=400, 
                             max_depth=10, 
                             min_samples_split=20, 
                             min_samples_leaf = 8, 
                             max_features = 'auto',                                 
                             class_weight= None,
                             bootstrap= True,
                             random_state=rand_st)
        sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
        print ('Wrapper Select - Random Forest: ')
        
        fit_mod=sel.fit(data_np, target_train)    
        sel_idx=fit_mod.get_support()

    if fs_type==3:                                                                ######Only work if the Target is binned###########
        #Univariate Feature Selection - Chi-squared
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

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np.columns)):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(data_np.columns[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np.columns), len(temp))
    print('\n')

'''                
    ##3) Filter selected columns from original dataset #########
    selected_features = list(data_np.columns[np.array(sel_idx).astype(bool)])                          #Deletes non-selected features by index)
    data_test = data_test[selected_features]
    data_np = data_np[selected_features]
'''