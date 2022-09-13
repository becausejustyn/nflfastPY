
### Air Yards Analysis
#https://www.fantasyfootballdatapros.com/blog/intermediate/27

'''
The plot at the end is a nice way to compare players
'''

import pandas as pd # for data munging
from matplotlib import pyplot as plt # for making visualizations
import seaborn as sns; sns.set_style('whitegrid'); # for making pretty visualizations
import numpy as np # to do number stuff
import requests # we're going to use this to make a HTTP request for player headshots
from io import BytesIO # we're going to use this to load image data
import nflfastpy


#load our 2021 play by play data
df = nflfastpy.load_pbp_data(year=2021)
# locate our team data with team colors
team_df = nflfastpy.load_team_logo_data()

# filter our original df to only include pass plays and those relevant passing columns
passing_df = df.loc[df['pass_attempt'] == 1, ['receiver_player_id', 'posteam', 'receiver_player_name', 'receiver_jersey_number',
                                              'air_yards', 'complete_pass', 'yards_gained', 'pass_touchdown']]

# find each plays PPR fantasy points scored for the receiver
passing_df['ppr_receiving_fantasy_points'] = passing_df['complete_pass'] + 6*passing_df['pass_touchdown'] + 0.1*passing_df['yards_gained']
# rename the color before we merge it with our passing_df
team_df = team_df.rename({'team_abbr': 'posteam'}, axis=1)
# merge the team data and the passing_df data
passing_df = passing_df.merge(team_df, how='left', on='posteam')

# find the top 10 players in terms of receiving relevant fantasy points scored
top_ppr_fantasy_points = passing_df[['receiver_player_id', 'ppr_receiving_fantasy_points']].groupby(
    'receiver_player_id', as_index=False).sum().sort_values(by='ppr_receiving_fantasy_points', ascending=False)[:10]

# filter our df to only include those plays that involved those players above
passing_df = passing_df.loc[passing_df['receiver_player_id'].isin(top_ppr_fantasy_points['receiver_player_id'])]
# drop rows that have na values. These are rows that don't have air yards data. 
passing_df = passing_df.dropna()
# filter for relevant columns
passing_df = passing_df[['receiver_player_id', 'receiver_player_name', 'air_yards', 'team_color', 'team_logo_wikipedia']]


passing_df['receiver_player_name'].unique()

#bring in roster data
roster_df = nflfastpy.load_roster_data(2021)

#change the player name column to have the same format as our passing_df one
roster_df['receiver_player_name'] = roster_df['full_name'].apply(lambda x: '.'.join([x.split()[0][0], x.split()[-1]]))

#filter our results a bit more so we get closer to our solution (join on id)
roster_df = roster_df.loc[
    (roster_df['season'] == 2021) & 
    (roster_df['gsis_id'].isin(passing_df['receiver_player_id'].unique())) & 
    (roster_df['position'].isin(['WR', 'TE', 'RB'])), 
    ['receiver_player_name', 'position', 'full_name', 'headshot_url']]

# reset the index to a range index 0 -> len(roster_df)
roster_df = roster_df.reset_index(drop=True)

# display all rows in our dataframe
with pd.option_context('display.max_rows', None):
    display(roster_df)
    
passing_df = passing_df.merge(roster_df, on='receiver_player_name', how='left').drop(['full_name', 'position'], axis=1)

groups = [group for group in passing_df.groupby('receiver_player_id')]

# 5 rows, 2 columns
fig, axes = plt.subplots(5, 2, figsize=(13, 15))
rows, columns = axes.shape[0], axes.shape[1]
i = 0

for row in range(rows):
    for col in range(columns):
        # get the df object
        player_df = groups[i][-1]
        player_name = player_df['receiver_player_name'].values[0]
        primary_color = player_df['team_color'].values[0]
        headshot_url = player_df['headshot_url'].values[0]
        # make a HTTP request to grab the player image
        response = requests.get(headshot_url)
        # load the image as bytecode
        img = plt.imread(BytesIO(response.content))
        # plot a KDE plot of the player's air yards on the row, col ax and color it with the player's team color
        ax = sns.kdeplot(player_df['air_yards'], ax=axes[row, col], label=player_name, color=primary_color)
        ax.set_xlim(-5)
        lines = ax.get_lines()[0].get_xydata()
        x, y = lines[:, 0], lines[:, 1]
        #extent argument left, right, bottom, top
        ax.imshow(img, extent=[75, 100, 0, 0.03], aspect='auto', zorder=1000)
        #fill the area underneath the curve
        ax.fill_between(x, y, color=primary_color, alpha=0.4)
        ax.set_xticks(range(0, 101, 10))
        ax.set_yticks(np.linspace(0, 0.07, 10))
        ax.set_xlabel('Air Yards')
        ax.set_ylabel('Density')
        #show us the legend
        ax.legend()
        i += 1

plt.tight_layout()