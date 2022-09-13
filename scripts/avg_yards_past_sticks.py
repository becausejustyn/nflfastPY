#Imports
import pandas as pd
import seaborn as sns
import numpy as np
import pyreadr

%config InlineBackend.figure_format = 'retina'
pd.set_option('mode.chained_assignment', None)

#load data
data1 = pyreadr.read_r('/Users/justynrodrigues/Documents/nfl/data/pbp/play_by_play_2021.rds')
data = data1[None]

df = data.loc[(data['season'] == 2021) & (data['down'] == 3) & (data['play_type'] == 'pass')]
df['past_the_sticks'] = 0
df.loc[(df['air_yards'] >= df['ydstogo']), 'past_the_sticks'] = 1
qbs = df.groupby(by = 'passer_player_name')['past_the_sticks', 'epa', 'ydstogo', 'third_down_converted'].mean()
qbs['attempts'] = df.groupby(by = 'passer_player_name')['air_yards'].count()
qbs = qbs.loc[qbs['attempts'] >= 100]

qbs.sort_values(by = 'past_the_sticks', ascending = False)

print('League Average is: ' + str(qbs['past_the_sticks'].mean()))