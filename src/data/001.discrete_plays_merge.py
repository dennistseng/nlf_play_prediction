# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:02:38 2020

@author: M44427


Just some additional preprocessing
"""

import numpy as np
import pandas as pd

plays = pd.read_csv("../../data/clean/plays.csv", low_memory = False)
drives = pd.read_csv("../../data/drives/drives.csv", low_memory = False)
drives.drop(['posteam_type', 'year', 'month', 'qtr', 'cd_start_yds_to_go', 'cd_start_time_left', 'cd_start_time_left', 'cd_start_score_diff',
             'cd_plays', 'points_scored', 'cd_run_percentage', 'cd_drive_length', 'cd_avg_yds_to_go_3rd', 'cd_avg_first_yds_gain', 'cd_net_penatly_yd_play',
             'cd_net_penalties_play', 'cd_pass_yard_att', 'cd_rush_yard_att', 'cd_sacks_play', 'cd_tfl_play', 'cd_expl_runs_play', 'cd_expl_pass_play',
             'cd_pass_completion_pct', 'cd_third_conversions'], axis = 1, inplace = True)

#del drives['ld_outcome']
#del drives['ld_opp_outcome']

# Get current drives outcome to remove any drives that end in turnovers or qb_kneels
current_drive_outcome = pd.read_csv("../../data/clean/DriveOutcome.csv", low_memory = False)
del current_drive_outcome['Unnamed: 0']
plays = pd.merge(plays, current_drive_outcome, how='inner', on=['game_id', 'drive', 'posteam'])
plays = plays[~((plays['drive_outcome'] == 'turnover') | (plays['drive_outcome'] == "turnover_on_downs") | (plays['drive_outcome'] == "qb_kneel") | (plays['drive_outcome'] == "qb_spike"))]
del plays['drive_outcome']

plays = pd.merge(plays, drives, how='inner', on=['game_id', 'drive', 'posteam'])

# Drop for plays that aren't relative in a plays persepective
plays.drop(['drive_starting_time', 'drive_starting_location', 'drive_end_time', 'drive_end_location'], axis = 1, inplace = True)


# Merge prior drive information
# Change certain columns to categorical variables
#drives['month'] = drives['month'].astype('category')
#drives['qtr'] = drives['qtr'].astype('category')


# Create Game-Drive ID
plays['game-drive'] = plays['game_id'].astype(str) + '-' + plays['drive'].astype(str)
#del plays['game_id']
del plays['drive']
#del plays['play_id']



# Set up for target variable
plays['next_play'] = plays['play_type'].shift(-1)
plays['next_id'] = plays['game-drive'].shift(-1)
plays['target'] = np.where(plays['next_id'] == plays['game-drive'], plays['next_play'], np.nan)

del plays['next_id']
del plays['next_play']
del plays['sp']
#del plays['play_type']



### remove unecessary columns
plays.drop(['fumble_forced', 'fumble_not_forced', 'fumble_out_of_bounds', 'safety', 'penalty', 'fumble_lost', 'extra_point_attempt',
            'two_point_attempt', 'field_goal_attempt', 'points_earned', 'RunOver10', 'PassOver20', 'complete_pass', 'touchdown', 'rush_touchdown', 'pass_touchdown', 
            'rush_attempt', 'pass_attempt', 'qb_hit', 'tackled_for_loss', 'incomplete_pass', 'qb_kneel'], axis =1, inplace = True)


# Remove in the interim
#del plays['posteam']
#del plays['GameYear']

del plays['defteam']
del plays['pass_length']
del plays['pass_location']
del plays['run_gap']
del plays['run_location']
del plays['Time']

del plays['interception']

    
# Create Dummy Data

del plays['game-drive']
#plays.to_csv('../../data/clean/model_plays_categorical.csv', index = False)

plays['home'] = pd.get_dummies(plays['posteam_type'], drop_first=True)
plays['game_half'] = pd.get_dummies(plays['game_half'], drop_first=True)
del plays['posteam_type']
plays = pd.get_dummies(plays, columns=['play_type', 'ld_outcome', 'ld_opp_outcome'])


# Remove last plays of drives
plays.dropna(inplace = True)

# Check if any columns have null values
drives.columns[drives.isna().any()].tolist()



# Double Check Output

plays.to_csv('../../data/clean/model_plays.csv', index = False)
