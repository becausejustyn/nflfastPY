
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

data = pd.concat(
    [nfl.load_pbp_data(season).assign(season = season) for season in range(1999, 2021)]
)

#if you do not want to install nflfastpy

# Read in data
YEAR = 2021

data = pd.read_csv(
  'https://github.com/guga31bb/nflfastR-data/blob/master/data/play_by_play_' + 
  str(YEAR) + '.csv.gz?raw=True',
  compression = 'gzip', low_memory = False)
  
data = r.data
data_df = data.copy()
print(data.head())

#View a random sample of our df to ensure everything is correct          
data.sample(3)

#The last step in preprocessing for this particular analysis is dropping null values to avoid jumps in our WP chart. To clean things up, we can filter the columns to show only those that are of importance to us.

cols = ['home_wp', 'away_wp', 'game_seconds_remaining']
data_df = data_df[cols].dropna()

#View new df to again ensure everything is correct
data_df

### Game Excitement Index  
## $$\frac{2400}{\text{Length of Game}} \sum_{i = 2}^{\text{n plays}} \mid \text{WinProb(i) - WinProb(i - 1)} \mid$$

## > the formula sums the absolute value of the win probability change from each play
## `https://sports.sites.yale.edu/game-excitement-index-part-ii`

#Calculate average length of 2019 games for use in our function
avg_length = data.groupby(by = ['game_id'])['epa'].count().mean()

def calc_gei(game_id):
  game = data[(data['game_id']==game_id)]
  #Length of game
  length = len(game)
  #Adjusting for game length
  normalize = avg_length / length
  #Get win probability differences for each play
  win_prob_change = game['home_wp'].diff().abs()
  #Normalization
  gei = normalize * win_prob_change.sum()
  return(gei)

print(f"Buffalo @ Chiefs GEI: {calc_gei('2021_20_BUF_KC')}")

#Set style
plt.style.use('dark_background')

#Create a figure
fig, ax = plt.subplots(figsize=(16,8))

#Generate lineplots
sns.lineplot('game_seconds_remaining', 'away_wp', 
             data=data_df, color='#3B47CC',linewidth=2)

sns.lineplot('game_seconds_remaining', 'home_wp', 
             data=data_df, color='#E31837',linewidth=2)

#Generate fills for the favored team at any given time

ax.fill_between(data_df['game_seconds_remaining'], 0.5, data_df['away_wp'], 
                where=data_df['away_wp']>.5, color = '#3B47CC',alpha=0.3)

ax.fill_between(data_df['game_seconds_remaining'], 0.5, data_df['home_wp'], 
                where=data_df['home_wp']>.5, color = '#E31837',alpha=0.3)
                
#Labels
plt.ylabel('Win Probability %', fontsize=16)
plt.xlabel('', fontsize=16)

#Divider lines for aesthetics
plt.axvline(x=900, color='white', alpha=0.7)
plt.axvline(x=1800, color='white', alpha=0.7)
plt.axvline(x=2700, color='white', alpha=0.7)
plt.axhline(y=.50, color='white', alpha=0.7)

#Format and rename xticks
ax.set_xticks(np.arange(0, 3601,900))

plt.gca().invert_xaxis()
x_ticks_labels = ['End','End Q3','Half','End Q1','Kickoff']
ax.set_xticklabels(x_ticks_labels, fontsize=12)

#Titles

#[Text(0, 0, 'End'), Text(900, 0, 'End Q3'), Text(1800, 0, 'Half'), Text(2700, 0, 'End Q1'), Text(3600, 0, 'Kickoff')]

plt.suptitle('Buffalo Bills @ Kansas City Chiefs', 
             fontsize=20, style='italic',weight='bold')

plt.title('KC 42, BUF 36 - Div Round ', fontsize=16, 
          style='italic', weight='semibold')

#Creating a textbox with GEI score
props = dict(boxstyle='round', facecolor='black', alpha=0.6)
plt.figtext(.133,.85,'Game Excitement Index (GEI): 7.29',style='italic',bbox=props)

#Citations
plt.figtext(0.131,0.137,'Data: @nflfastR')

plt.show()
