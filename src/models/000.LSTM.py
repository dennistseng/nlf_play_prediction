# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:36:58 2020

@author: halfc
"""

import numpy as np
import pandas as pd

import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Embedding, Lambda, TimeDistributed, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler

# Load Data
data = pd.read_csv("../../data/clean/plays.csv", low_memory = False)

# Remove in the interim
del data['posteam']
del data['defteam']
del data['game_half']
del data['pass_length']
del data['pass_location']
del data['run_gap']
del data['run_location']
del data['Time']

# Dummy Data
data['home'] = pd.get_dummies(data['posteam_type'], drop_first=True)
del data['posteam_type']
data['playtype'] = data['play_type']
data = pd.get_dummies(data, columns=['playtype'])

# Create Game-Drive ID
data['game-drive'] = data['game_id'].astype(str) + '-' + data['drive'].astype(str)
del data['game_id']
del data['drive']
del data['play_id']

# Set up for target variable
data['next_play'] = data['play_type'].shift(-1)
data['next_id'] = data['game-drive'].shift(-1)
data['target'] = np.where(data['next_id'] == data['game-drive'], data['next_play'], np.nan)

del data['next_id']
del data['next_play']
del data['sp']
del data['play_type']

# Remove last plays of drives
data.dropna(inplace = True)


# Normalize Data
scalerx = MinMaxScaler( feature_range=(0, 1) )
num_cols = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
data_scale = data.copy()
data_scale[num_cols] = scalerx.fit_transform(data_scale[num_cols])

# Create Target Variable
target = data_scale.iloc[:, -2:]
del data_scale['target']
target['target'] = target['target'].astype('category').cat.codes

# Check for null values
data.isnull().any()
data_scale.isnull().any()

#drive_format = np.array(list(data.groupby('game-drive', as_index=True).apply(pd.DataFrame.as_matrix)))
drive_format = np.array(list(data_scale.groupby('game-drive', as_index=True).apply(pd.DataFrame.to_numpy)))
target_format = np.array(list(target.groupby('game-drive', as_index=True).apply(pd.DataFrame.to_numpy)))

#drive_format = data.groupby('game-drive').agg(list)
#drive_format = data.groupby('game-drive').data.apply(list).reset_index()
#data.groupby('game-drive').data.apply(np.array) 
#aaa = np.array(list(data.groupby('game-drive', as_index=True).values))
#a = np.delete(drive_format, 50, 2)
#a = drive_format[: , :, 0:49]

# Iterate over all drives to remove grouping ID
#drive_format[0] = np.array(pd.DataFrame(drive_format[0][:,0:56]))

count = 0
for i in drive_format:
    drive_format[count] = np.array(pd.DataFrame(drive_format[count][:,0:56]))
    count += 1

count = 0
for i in target_format:
    target_format[count] = np.array(pd.DataFrame(target_format[count][:,1:]).astype('float64'))
    count += 1

'''
# Normalize Data
scalerx = MinMaxScaler( feature_range=(0, 1) )
#data_scale = pd.DataFrame(scalerx.fit_transform(data[data.columns[0:55]]), columns = data.columns[0:55])
data_scale = pd.DataFrame(scalerx.fit_transform(drive_format))
data_scale['game-drive'] = data['game-drive']
data_scale['target'] = data['target']
'''

import keras
padded_inputs = keras.preprocessing.sequence.pad_sequences(
    drive_format, padding="post"
)

padded_outputs = keras.preprocessing.sequence.pad_sequences(
    target_format, padding="post"
)

padded_inputs = padded_inputs.astype(float)
padded_outputs = padded_outputs.astype(float)

#a = tensorflow.convert_to_tensor(target_format, dtype=tensorflow.float64)
    
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(25, 56)))
model.add(LSTM(50, activation='relu', return_sequences = True))
model.add(TimeDistributed(Dense(6)))
#model.add(Dense(6))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy']) 
model.summary()

model.fit(padded_inputs, padded_outputs, epochs=10, verbose=1)



model2 = Sequential()
model2.add(Masking(mask_value=0., input_shape=(25, 56)))
model2.add(LSTM(128, activation='relu', return_sequences = True))
model2.add(LSTM(64, activation='relu', return_sequences = True))
model2.add(LSTM(64, activation='relu', return_sequences = True))
model2.add(LSTM(32, activation='relu', return_sequences = True))
model2.add(TimeDistributed(Dense(6, activation = 'softmax')))
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy']) 
model2.summary()

model2.fit(padded_inputs, padded_outputs, epochs=10, verbose=1)

'''
# plot history
pyplot.plot(model.history['loss'], label='train')
pyplot.plot(model.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
'''