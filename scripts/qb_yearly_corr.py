
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import glob
all_files = glob.glob('/Users/justynrodrigues/Documents/nfl/data/pbp/csv/*.csv.gz')
data = pd.concat((pd.read_csv(__, low_memory=False, index_col=0) for __ in all_files))
#Variables we care about
variables = ['epa', 'air_yards', 'cpoe', 'wpa']

#Groupby passer name, season, team
qbs = data.groupby(by=['passer_player_name', 'season', 'posteam'])[variables].mean()
#Get a couple of other variables that aren't averages
qbs['std_epa'] = data.groupby(by=['passer_player_name', 'season', 'posteam'])['epa'].std()
qbs['attempts'] = data.groupby(by=['passer_player_name', 'season', 'posteam'])['epa'].count()
#Only use QBs with over 128 attempts in a year (NGS uses this cutoff)
qbs = qbs.loc[qbs['attempts']>=128]
#reset the index so passer_player_name is a column and not an index
qbs.reset_index(inplace=True)
#Only get QBs who have more than one season of data (or there's nothing to correlate!)
qbs = qbs[qbs.groupby('passer_player_name').passer_player_name.transform('count') > 1]

#Shift the QB variables by 1 row for each QB
lqbs = qbs.groupby(by='passer_player_name').shift(-1)
#rename QB columns so we can join them
qbs.columns = ['passer_player_name','prev_season',
               'prev_posteam','prev_epa',
               'prev_air_yards',
               'prev_cpoe',
               'prev_wpa',
               'prev_std_epa',
               'prev_attempts']
#Join the lagged variables with the original dataframe
new_qbs = pd.concat((qbs,lqbs),axis=1).dropna(subset=['season'])

corr = new_qbs.corr(method='pearson')
#No same year correlations with this view, 
#just how variables in a previous year correlates to variables in the next year
corr.iloc[0:7,7:]

#Plotting
plt.style.use('seaborn-talk')
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap='RdBu',
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()