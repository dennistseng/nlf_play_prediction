"""
Clean Module for NFL Play Prediction Project
@author: Dennis Tseng

Part of the DSC 672 Capstone Final Project Class
Group 3: Dennis Tseng, Scott Elmore, Dongmin Sun

"""

#import libraries
import numpy as np
import pandas as pd

# %% 
#######################################################
## Load and Clean Play By Play Dataset
#######################################################

# Load datasets
raw_play_by_play = pd.read_csv("../../data/raw/ConsolidateOutput.csv", low_memory = False)
nfl_weather = pd.read_csv("../../data/raw/Weather_NFL.csv", low_memory = False)

######
# Remove any colums deemed unecessary for both pre-processing and analysis 
######

# Drop any columns/features that are team or player specific
raw_play_by_play.drop(['fumble_recovery_1_player_id', 'fumble_recovery_1_team', 'fumble_recovery_1_yards', 'fumble_recovery_1_player_name', 'fumble_recovery_2_player_id', 'fumble_recovery_2_player_name', 
                    'fumble_recovery_2_team', 'fumble_recovery_2_yards', 'fumbled_1_team', 'fumbled_1_player_name', 'fumbled_1_player_id', 'fumbled_2_player_id', 'fumbled_2_player_name', 'fumbled_2_team', 
                    'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_defense_2_player_id', 'pass_defense_2_player_name', 'assist_tackle_4_player_id', 'assist_tackle_4_player_name', 
                    'assist_tackle_4_team', 'assist_tackle_3_team', 'assist_tackle_3_player_name', 'assist_tackle_3_player_id', 'assist_tackle_2_player_id', 'assist_tackle_2_player_name', 
                    'assist_tackle_2_team', 'assist_tackle_1_player_id', 'assist_tackle_1_player_name', 'assist_tackle_1_team', 'solo_tackle_2_player_name', 'solo_tackle_1_player_name', 
                    'solo_tackle_2_team', 'solo_tackle_1_team', 'solo_tackle_1_player_id', 'solo_tackle_2_player_id', 'forced_fumble_player_2_player_name', 'forced_fumble_player_2_player_id',
                    'forced_fumble_player_2_team', 'forced_fumble_player_1_player_name', 'forced_fumble_player_1_player_id', 'forced_fumble_player_1_team', 'qb_hit_2_player_name', 'qb_hit_2_player_id',
                    'qb_hit_1_player_id', 'qb_hit_1_player_name', 'tackle_for_loss_2_player_name', 'tackle_for_loss_2_player_id', 'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name',
                    'penalty_player_name', 'penalty_player_id', 'blocked_player_id', 'blocked_player_name', 'own_kickoff_recovery_player_id', 'own_kickoff_recovery_player_name', 'own_kickoff_recovery_td',
                    'kicker_player_name', 'kicker_player_id', 'punter_player_name', 'punter_player_id', 'lateral_kickoff_returner_player_id', 'lateral_kickoff_returner_player_name', 'kickoff_returner_player_id',
                    'kickoff_returner_player_name', 'lateral_punt_returner_player_id', 'lateral_punt_returner_player_name', 'punt_returner_player_id', 'punt_returner_player_name', 'lateral_interception_player_id',
                    'lateral_interception_player_name', 'interception_player_id', 'interception_player_name', 'lateral_sack_player_id', 'lateral_sack_player_name', 'lateral_rusher_player_name', 'lateral_rusher_player_id',
                    'rusher_player_name', 'rusher_player_id', 'lateral_receiver_player_id', 'lateral_receiver_player_name', 'passer_player_id', 'passer_player_name', 'receiver_player_id', 'receiver_player_name'
                    ], axis = 1, inplace = True)


# Drop any columns that provide wpa (win probability added) or epa (expected points added) features.
raw_play_by_play.drop(['total_away_raw_yac_wpa', 'total_home_raw_yac_wpa', 'total_away_raw_air_wpa', 'total_home_comp_air_wpa', 'total_away_comp_air_epa', 'total_away_comp_air_wpa', 'total_away_comp_yac_epa', 
                    'total_away_comp_yac_wpa', 'total_home_epa', 'total_away_pass_epa', 'total_away_pass_wpa', 'total_away_raw_air_epa', 'total_away_raw_yac_epa', 'total_away_epa', 'total_away_rush_epa', 
                    'total_away_rush_wpa', 'total_home_comp_air_epa', 'total_home_comp_yac_epa', 'total_home_rush_wpa', 'total_home_pass_epa', 'total_home_pass_wpa', 'total_home_comp_yac_wpa', 
                    'total_home_raw_air_epa', 'total_home_raw_yac_epa', 'total_home_rush_epa', 'comp_yac_wpa', 'comp_yac_epa', 'comp_air_epa', 'comp_air_wpa', 'no_score_prob', 'opp_fg_prob', 'opp_safety_prob', 
                    'opp_td_prob', 'air_epa', 'yac_epa', 'air_wpa', 'epa', 'ep', 'wp', 'wpa', 'fg_prob', 'safety_prob', 'td_prob', 'extra_point_prob', 'two_point_conversion_prob', 'def_wp', 'home_wp', 'away_wp', 
                    'home_wp_post', 'yac_wpa', 'away_wp_post', 'total_home_raw_air_wpa'], axis = 1, inplace = True)


# Drop any columns pertaining to kickoffs and punts
raw_play_by_play.drop(['kick_distance', 'kickoff_inside_twenty', 'kickoff_attempt', 'kickoff_downed', 'kickoff_in_endzone', 'kickoff_fair_catch', 'kickoff_out_of_bounds', 
                    'own_kickoff_recovery', 'punt_attempt', 'punt_inside_twenty', 'punt_downed', 'punt_in_endzone', 'punt_out_of_bounds', 'punt_fair_catch', 'return_team', 
                    'return_yards', 'return_touchdown'], axis = 1, inplace = True)


# Drop variables for columns that are related to specific actions like penalties, challenges, first/4th downs
raw_play_by_play.drop(['first_down_pass', 'first_down_penalty', 'first_down_rush', 'field_goal_result', 'third_down_failed', 'fourth_down_converted', 'fourth_down_failed', 
                    'solo_tackle', 'replay_or_challenge', 'replay_or_challenge_result', 'defensive_extra_point_attempt', 'defensive_extra_point_conv', 'lateral_recovery', 'lateral_reception', 'lateral_rush', 
                    'lateral_return', 'timeout_team', 'td_team', 'assist_tackle', 'touchback', 'penalty_type', 'extra_point_result', 'two_point_conv_result',
                    'defensive_two_point_attempt'], axis = 1, inplace = True)


# Drop some game-situation specific columns
raw_play_by_play.drop(['home_timeouts_remaining', 'away_timeouts_remaining', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'total_home_score', 'total_away_score', 'posteam_score', 
                    'defteam_score', 'posteam_score_post', 'defteam_score_post', 'yrdln', 'half_seconds_remaining', 'quarter_end', 'time', 'goal_to_go', 
                    'quarter_seconds_remaining', 'side_of_field', 'away_team', 'home_team'], axis = 1, inplace = True)


# Remove any duplicate rows
raw_play_by_play.drop_duplicates(keep='first', inplace=True)


# %% 
#######################################################
## Merge Weather, Madden, and Betting Data
#######################################################

# Clean weather and aggregate weather data to merge to play-by-play data
nfl_weather = nfl_weather.groupby(['game_id']).agg({'Rain': np.min, 'Snow': np.min, 'Wind':np.min, 'Fog':np.min,
                                                    'Dome':np.mean, 'Time':np.min})
raw_play_by_play = pd.merge(raw_play_by_play, nfl_weather, how='inner', on=['game_id'])

# Merge Madden data

# Merge betting and probability data

# %% 
#######################################################
## Clean values within columns
#######################################################

# Replace any null/None/NaN values
raw_play_by_play['desc'].fillna('Other' , inplace=True)
raw_play_by_play['play_type'].fillna('Other' , inplace=True)
raw_play_by_play['down'].fillna(0, inplace=True)
raw_play_by_play['yards_gained'].fillna(0, inplace=True)
raw_play_by_play['penalty_yards'].fillna(0, inplace=True)
raw_play_by_play['game_seconds_remaining'].fillna(-1, inplace=True)
raw_play_by_play['two_point_attempt'].fillna(0, inplace=True)
raw_play_by_play['defensive_two_point_conv'].fillna(0, inplace=True)
raw_play_by_play['score_differential'].fillna(0, inplace=True)
raw_play_by_play['score_differential_post'].fillna(0, inplace=True)
raw_play_by_play['incomplete_pass'].fillna(0, inplace=True)
raw_play_by_play['timeout'].fillna(0, inplace=True)
raw_play_by_play['yards_after_catch'].fillna(0, inplace=True)
raw_play_by_play['air_yards'].fillna(0, inplace=True)

# Set description strings to lowercase
raw_play_by_play['desc'] = raw_play_by_play['desc'].str.lower()

# Calculate/determine if a play resulted in points and fixes errors within data (defensive 2 point scores)
raw_play_by_play['points_earned'] = raw_play_by_play['score_differential_post'] - raw_play_by_play['score_differential']
raw_play_by_play['points_earned'] = np.where(raw_play_by_play['defensive_two_point_conv'] == 1, -2, raw_play_by_play['points_earned'])
raw_play_by_play['points_earned'].fillna(0, inplace=True)
del raw_play_by_play['score_differential_post']

raw_play_by_play['play_type'] = np.where(raw_play_by_play['two_point_attempt'] == 1, "2pt Attempt", raw_play_by_play['play_type'])
raw_play_by_play['points_earned'] = np.where(raw_play_by_play['incomplete_pass'] == 1, 0, raw_play_by_play['points_earned'])

# Convert dates to usable months and year attributes
raw_play_by_play['GameMonth'] = pd.DatetimeIndex(raw_play_by_play['game_date']).month
raw_play_by_play['GameYear'] = pd.DatetimeIndex(raw_play_by_play['game_date']).year
del raw_play_by_play['game_date']

# %% 
#######################################################
## Time Stoppages and Non(important)-Plays
#######################################################

# Separate time stoppages (end of quarters, end of games, timeouts)
timeouts = raw_play_by_play[(raw_play_by_play['timeout'] == 1) & (raw_play_by_play['desc'].str.contains('timeout') & raw_play_by_play['down'] == 0)]
timeouts = timeouts[['game_id','drive','play_id']]
timeouts = timeouts.groupby(['game_id', 'drive']).agg(['count'])

playStoppages = raw_play_by_play[(raw_play_by_play['desc'].str.contains('end of quarter')) | (raw_play_by_play['desc'].str.contains('end quarter')) | (raw_play_by_play['desc'].str.contains('two-minute warning'))]
playStoppages = playStoppages[['game_id','drive','play_id']]
playStoppages = playStoppages.groupby(['game_id', 'drive']).agg(['count'])
endOfGames = raw_play_by_play[(raw_play_by_play['desc'].str.contains('end game')) | (raw_play_by_play['desc'].str.contains('end of game'))]

# Additional Cleanup for offsetting penalties (no plays), substitutions, other non-play tuples
misc = raw_play_by_play[(raw_play_by_play['play_type'] == "Other") & (raw_play_by_play['game_seconds_remaining'] == -1)]
offset = raw_play_by_play[(raw_play_by_play['desc'].str.contains('offsetting')) & (raw_play_by_play['play_type'] == "Other")]
misc2 = raw_play_by_play[(raw_play_by_play['play_type'] == "Other")]
misc = pd.concat([misc,misc2])
del misc2

# Remove Timeouts and Play Stoppages (Non Penalties) from Play Set
raw_play_by_play = raw_play_by_play[~((raw_play_by_play['timeout'] == 1) & (raw_play_by_play['play_type'] == "no_play") & (raw_play_by_play['down'] == 0))]
raw_play_by_play = raw_play_by_play[~((raw_play_by_play['down'] == 0) & (raw_play_by_play['desc'].str.contains('timeout')))]
raw_play_by_play = raw_play_by_play[~raw_play_by_play['desc'].str.contains('two-minute warning', na=False)]
raw_play_by_play = raw_play_by_play[~((raw_play_by_play['desc'].str.contains('end of quarter')) | (raw_play_by_play['desc'].str.contains('end quarter'))| (raw_play_by_play['desc'].str.contains('end of half')))]
raw_play_by_play = raw_play_by_play[~((raw_play_by_play['desc'].str.contains('end game')) | (raw_play_by_play['desc'].str.contains('end of game')))]
raw_play_by_play = raw_play_by_play[~((raw_play_by_play['play_type'] == "Other") & (raw_play_by_play['game_seconds_remaining'] == -1))]
raw_play_by_play = raw_play_by_play[~((raw_play_by_play['play_type'] == "Other") & (raw_play_by_play['desc'].str.contains('offsetting')))]
raw_play_by_play = raw_play_by_play[~(raw_play_by_play['play_type'] == "Other")]
raw_play_by_play = raw_play_by_play[~(raw_play_by_play['play_type'] == "extra_point")]
raw_play_by_play = raw_play_by_play[~(raw_play_by_play['play_type'] == "2pt Attempt")]

# Remove Kickoffs, as they don't count as the first play of a drive and would affect
# starting field position calculations
raw_play_by_play = raw_play_by_play[~(raw_play_by_play['play_type'] == "kickoff")]

# Take out plays labeled qb_kneels that are at the begining of the drive (touchbacks)
idx = raw_play_by_play.groupby(['game_id', 'drive', 'posteam'])['game_seconds_remaining'].idxmax()
Touchbacks = raw_play_by_play.loc[idx]
Touchbacks = Touchbacks[(Touchbacks['play_type'] == 'qb_kneel')]
raw_play_by_play = raw_play_by_play[~raw_play_by_play.index.isin(Touchbacks.index)]


# %%
#######################################################
## Build tables to join back to Drive dataset later
#######################################################
ThirdDown = raw_play_by_play[raw_play_by_play['down'] == 3]
ThirdDown = ThirdDown[['game_id', 'drive', 'posteam', 'ydstogo','third_down_converted']]
ThirdDown = ThirdDown.groupby(['game_id', 'drive', 'posteam']).agg({'ydstogo' : np.mean, 'third_down_converted' : np.sum})

FirstDown = raw_play_by_play[raw_play_by_play['down'] == 1]
FirstDown = FirstDown[['game_id', 'drive', 'posteam', 'yards_gained']]
FirstDown = FirstDown.groupby(['game_id', 'drive', 'posteam']).agg({'yards_gained' : np.mean})

PenaltiesPos = raw_play_by_play[((raw_play_by_play['penalty'] == 1) & (raw_play_by_play['posteam'] == raw_play_by_play['penalty_team']))]
PenaltiesPos = PenaltiesPos[['game_id', 'drive', 'posteam', 'penalty', 'penalty_yards']]
PenaltiesPos = PenaltiesPos.groupby(['game_id', 'drive', 'posteam']).agg({'penalty_yards' : np.sum, 'penalty' : np.sum})

PenaltiesDef = raw_play_by_play[((raw_play_by_play['penalty'] == 1) & (raw_play_by_play['posteam'] != raw_play_by_play['penalty_team']))]
PenaltiesDef = PenaltiesDef[['game_id', 'drive', 'posteam', 'penalty', 'penalty_yards']]
PenaltiesDef = PenaltiesDef.groupby(['game_id', 'drive', 'posteam']).agg({'penalty_yards' : np.sum, 'penalty' : np.sum})

# Remove 'no play' plays that were needed in case they were penalties, but not for actual drives. Needed in case they're the first or last play of drives
raw_play_by_play = raw_play_by_play[~(raw_play_by_play['play_type'] == "no_play")]

# Take final plays 
idx = raw_play_by_play.groupby(['game_id', 'drive', 'posteam'])['game_seconds_remaining'].idxmin()
FinalPlays = raw_play_by_play.loc[idx]

# Take out punts and field goals only if they're the last play in the drive
PuntsAndFG = FinalPlays[((FinalPlays['play_type'] == 'field_goal') | (FinalPlays['play_type'] == 'punt'))]
FG = PuntsAndFG[PuntsAndFG['play_type'] == "field_goal"]
FG = FG[['game_id','drive','posteam','points_earned']]
FG['points_earned'] = 3
FG.drop_duplicates(keep='first')

# Determine the final outcomes (not plays) for every drive
FinalPlays['drive_outcome'] = np.where(FinalPlays['play_type'] == "qb_kneel", 'qb_kneel', FinalPlays['play_type'])
FinalPlays['drive_outcome'] = np.where(((FinalPlays['drive_outcome'] == 'run') & (FinalPlays['touchdown'] == 1)), 'touchdown', FinalPlays['drive_outcome']) 
FinalPlays['drive_outcome'] = np.where(((FinalPlays['drive_outcome'] == 'pass') & (FinalPlays['touchdown'] == 1)), 'touchdown', FinalPlays['drive_outcome']) 
FinalPlays['drive_outcome'] = np.where((((FinalPlays['drive_outcome'] == 'pass') | (FinalPlays['drive_outcome'] == 'run')) &  ((FinalPlays['interception'] == 1) | (FinalPlays['fumble_lost'] == 1))), 'turnover', FinalPlays['drive_outcome']) 
FinalPlays['drive_outcome'] = np.where(((FinalPlays['drive_outcome'] == 'pass') | (FinalPlays['drive_outcome'] == 'run')) , 'turnover_on_downs', FinalPlays['drive_outcome']) 

DriveOutcome = FinalPlays[['game_id', 'drive','posteam','drive_outcome']]

# %%
#######################################################
## Create new features for analysis
#######################################################

# Explosive plays and penalties
raw_play_by_play['RunOver10'] = np.where((raw_play_by_play['play_type'] == "run") & (raw_play_by_play['yards_gained'] >= 10), 1, 0)
raw_play_by_play['PassOver20'] = np.where((raw_play_by_play['play_type'] == "pass") & (raw_play_by_play['yards_gained'] >= 20), 1, 0)
raw_play_by_play['penalty_yards'] = np.where((raw_play_by_play['play_type'] == "pass") & (raw_play_by_play['yards_gained'] >= 20), 1, 0)

# Create first play table to determine starting field position/yards to go
idx = raw_play_by_play.groupby(['game_id','drive', 'posteam'])['game_seconds_remaining'].idxmax()
FirstPlays = raw_play_by_play.loc[idx, ['game_id', 'drive', 'posteam', 'game_seconds_remaining', 'yardline_100']]
raw_play_by_play = pd.merge(raw_play_by_play, FirstPlays, how='left', on=['game_id', 'drive', 'posteam'])
raw_play_by_play.rename(columns={'game_seconds_remaining_x':'game_seconds_remaining', 'game_seconds_remaining_y':'drive_starting_time', 'yardline_100_x':'yardline_100', 'yardline_100_y':'drive_starting_location'}, inplace=True)

# Create last play table to determine time elapsed per drive
l_idx = raw_play_by_play.groupby(['game_id','drive', 'posteam'])['game_seconds_remaining'].idxmin()
LastPlays = raw_play_by_play.loc[l_idx, ['game_id', 'drive', 'posteam', 'game_seconds_remaining', 'yardline_100']]
raw_play_by_play = pd.merge(raw_play_by_play, LastPlays, how='left', on=['game_id', 'drive', 'posteam'])
raw_play_by_play.rename(columns={'game_seconds_remaining_x':'game_seconds_remaining', 'game_seconds_remaining_y':'drive_end_time', 'yardline_100_x':'yardline_100', 'yardline_100_y':'drive_end_location'}, inplace = True)


# %%
#######################################################
## This outputs play by play outcomes for use in creating historical features
#######################################################

prior_play_by_play = raw_play_by_play[['play_id','game_id', 'posteam', 'defteam', 'drive', 'sp', 'qtr', 'down', 'ydstogo', 'ydsnet','game_seconds_remaining', 'play_type', 'yards_gained', 'punt_blocked', 'interception', 'fumble_lost', 'points_earned']]
last_play_idx = prior_play_by_play.groupby(['game_id', 'drive', 'posteam'])['game_seconds_remaining'].idxmin()
last_plays = prior_play_by_play.loc[last_play_idx]

# set up 'Outcome' column
last_plays['drive_outcome'] = str(0)
last_plays['drive_outcome'] = np.where(last_plays['punt_blocked'] == 1, 'punt_blocked', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['fumble_lost'] == 1, 'fumble_lost', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['interception'] == 1, 'interception', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['play_type'] == 'punt', 'punt', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['play_type'] == 'field_goal', 'field_goal', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['points_earned'] == 6, 'touchdown', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(((last_plays['down'] == 4) & (last_plays['drive_outcome'] == '0')), 'turnover_on_downs', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['drive_outcome'] == '0', 'end_of_half', last_plays['drive_outcome'])

last_plays = last_plays[['game_id','posteam','drive','drive_outcome']]


# Finally, removes punts and field goals from drives for 'clean' play-by-play
raw_play_by_play = raw_play_by_play[~raw_play_by_play.index.isin(PuntsAndFG.index)]

# %%
#######################################################
## Last features for analysis
#######################################################

'''
For some reason moving these to the first create features cell leads to some weird issues during model training, so we keep this here
'''

# YPA for Rushes and Passes
Passes = raw_play_by_play[(raw_play_by_play['play_type']== "pass")].groupby(['game_id', 'drive', 'posteam']).agg({'yards_gained': np.sum})
Passes.rename({'yards_gained': 'PassYardage'}, axis=1, inplace=True)

Runs = raw_play_by_play[(raw_play_by_play['play_type']== "run")].groupby(['game_id', 'drive', 'posteam']).agg({'yards_gained': np.sum})
Runs.rename({'yards_gained': 'RunYardage'}, axis=1, inplace=True)

DrivePlays = raw_play_by_play[['game_id', 'drive','posteam','play_id']].groupby(['game_id', 'drive', 'posteam']).count()
DrivePlays.rename({'play_id': 'Count'}, axis=1, inplace=True)


# %%
#######################################################
## Remove features used above but not necessary for model training
#######################################################

del raw_play_by_play['penalty_team']
del raw_play_by_play['penalty_yards']
del raw_play_by_play['desc']
del raw_play_by_play['defensive_two_point_conv']
del raw_play_by_play['timeout']
del raw_play_by_play['third_down_converted']
del raw_play_by_play['punt_blocked']


# %%
#######################################################
## Outputs data for next stage/script
#######################################################

raw_play_by_play.to_csv('../../data/clean/plays.csv', index = False)
last_plays.to_csv('../../data/clean/last_plays.csv', index = False)
FG.to_csv('../../data/clean/FG.csv', index = False)
PenaltiesDef.to_csv('../../data/clean/PenaltiesDef.csv')
PenaltiesPos.to_csv('../../data/clean/PenaltiesPos.csv')
FirstDown.to_csv('../../data/clean/FirstDown.csv')
ThirdDown.to_csv('../../data/clean/ThirdDown.csv')
Runs.to_csv('../../data/clean/Runs.csv')
Passes.to_csv('../../data/clean/Passes.csv')
DriveOutcome.to_csv('../../data/clean/DriveOutcome.csv')