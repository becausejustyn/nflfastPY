
### Plotting Yards After Catch
# https://www.fantasyfootballdatapros.com/blog/intermediate/18

#https://www.fantasyfootballdatapros.com/blog/intermediate/18

import pandas as pd
import nflfastpy as nfl
from matplotlib import pyplot as plt
import seaborn as sns
import requests
from io import BytesIO

#grab nfl play by play data for 2020
df = nfl.load_pbp_data(2020)

#grab roster data which contains player positions and headshot_urls
#positions could be useful if you want to display only RBs, WRs, TEs, for example.
#we'll merge the roster data with the play by play data but not directly use it in this post.
roster = nfl.load_2020_roster_data()
#team colors for our plots
team_logo_colors = nfl.load_team_logo_data()

# filter our DataFrame to include only passing plays AND plays where the receiver_player_id column
# is not null
df = df.loc[(df['pass_attempt'] == 1) & (df['receiver_player_id'].notnull()), ['receiver_player_id', 'receiver_player_name', 'posteam', 'complete_pass', 'air_yards', 'yards_gained']]

# create a new ID column which can be merged with roster data
df['gsis_id'] = df['receiver_player_id'].apply(nfl.utils.convert_to_gsis_id)


#assign YAC yardage if a play results in a catch
# if no catch is made, then no YAC yardage occured
def assign_yac(row):
    if row['complete_pass']:
        return row['yards_gained'] - row['air_yards']
    else:
        return 0

#assign new column
df['yac'] = df.apply(assign_yac, axis=1)

#filter out unecessary columns
df = df[['gsis_id', 'receiver_player_name', 'posteam', 'yac']]

#merge roster data and team color data
df = df.merge(roster[['gsis_id', 'position', 'headshot_url']], on='gsis_id')
df = df.merge(team_logo_colors[['team_abbr', 'team_color', 'team_color2']].rename(columns={'team_abbr': 'posteam'}), on='posteam')

#run a aggregation to find the top 10 receivers this year in terms of air yards
df.groupby(['gsis_id', 'receiver_player_name', 'posteam', 'position'], as_index=False)['yac'].sum().sort_values(by='yac', ascending=False).head()


# top 10 players in terms of YAC yardage this season
top_10 = df.groupby(['gsis_id', 'receiver_player_name'], as_index=False)['yac'].sum().sort_values(by='yac', ascending=False)[:10]

df = df.loc[df['gsis_id'].isin(top_10['gsis_id'])]

#create figure and axes
#5 here means 5 rows
#2 here means 2 columns
#set the figsize to 10 inches x 15 inches
fig, ax = plt.subplots(5, 2, figsize=(10, 15))

#the ax object is a list of lists
#this is a function that "flattens" the list of lists in to a single list
flatten = lambda t: [item for sublist in t for item in sublist]

#flatten our list of axis lists
axes = flatten(ax)

#separate our play by play df in to 10 seperate dfs
players = [group[-1] for group in df.groupby('gsis_id')]

#zip that ^ and the axes together and iterate over them
for ax, player_df in zip(axes, players):

    #plot YAC distribution
    sns.kdeplot(player_df['yac'], ax=ax, lw=4, color=player_df['team_color'].values[0])

    #fill in the area underneath the curve
    xy = ax.get_lines()[0].get_xydata()
    x, y = xy[:, 0], xy[:, 1]
    ax.fill_between(x, y, color=player_df['team_color'].values[0], alpha=0.4)

    #adjust y ticks to only include these values
    ax.set_yticks([0, 0.1, 0.2])

    #set the axis title
    receiver_player_name = player_df['receiver_player_name'].values[0]
    ax.set_title(f'\n{receiver_player_name}', fontsize=16, fontweight=450)

    #set ylim and xlim
    ax.set_ylim(bottom=0, top=0.20)
    ax.set_xlim(left=-5, right=55)

    #remove the legend
    ax.get_legend().remove()

    #plot headshot image
    res = requests.get(player_df['headshot_url'].values[0])
    img = plt.imread(BytesIO(res.content))
    ax.imshow(img, extent=[40, 58, 0, 0.1], aspect='auto')

#set the title for the figure (not the axis's)
fig.suptitle('YAC distributions for top 10 YAC receivers 2020', fontsize=18)
#add some margin between axis's
fig.tight_layout()
#adjust the subplots to add some room for the super title
fig.subplots_adjust(top=0.92)
#set figure background title
fig.set_facecolor('white')