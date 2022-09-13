
import pandas as pd
import numpy as np
import os
import glob
import math
import gc
gc.collect()

import seaborn as sns
import matplotlib.pyplot as plt

source_folder = '/Users/justynrodrigues/Documents/nfl/data/pbp/csv/'
all_files = glob.glob('/Users/justynrodrigues/Documents/nfl/data/pbp/csv/*.csv.gz')

df = pd.concat((pd.read_csv(__, low_memory=False, index_col=0) for __ in all_files))

'''
seasons = list(range(2009,2019)) 

# merge seasons into one df
df = pd.DataFrame()
for season in seasons:
    path = './../data/nflscrapR-data/play_by_play_data/regular_season/reg_pbp_'+str(season)+'.csv'
    season_df = pd.read_csv(path)
    season_df['season'] = season
    df = pd.concat([df, season_df], axis=0)
'''


# standardize jaguars abbrev
df['posteam']=df['posteam'].replace('JAC','JAX')
# drop redzone plays & inside own 5
df = df.loc[(df.yardline_100 > 20) & (df.yardline_100 < 95)]
# drop extreme win probabilities (about 4% of plays)
df = df.loc[(df['wp']>=0.05) | (df['wp']<=0.95)]
# drop afc and nfc
conf_teams = ['AFC','NFC','IRV','APR','NPR','RIC','SANS']
df = df.loc[~df['posteam'].isin(conf_teams)]

### Load Roster Data
#found here: https://github.com/btatkinson/pfr_scraper

path = './../data/roster/roster_info.csv'
roster = pd.read_csv(path)
roster['sea_id'] = roster['season'].astype(str)+roster['player_id']
roster = roster[['id','player_id','season','pos','sea_id','name','age','height','weight','av']]

# qb heatmap
qbs = df.copy()

# drop unnecessary columns
qbs = qbs.drop(columns=['assist_tackle', 'assist_tackle_1_player_id', 'assist_tackle_1_player_name', 'assist_tackle_1_team', 'assist_tackle_2_player_id', 'assist_tackle_2_player_name', 'assist_tackle_2_team', 'assist_tackle_3_player_id', 'assist_tackle_3_player_name', 'assist_tackle_3_team', 'assist_tackle_4_player_id', 'assist_tackle_4_player_name', 'assist_tackle_4_team',
                       'kick_distance', 'kicker_player_id', 'kicker_player_name', 'kickoff_attempt', 'kickoff_downed', 'kickoff_fair_catch', 'kickoff_in_endzone', 'kickoff_inside_twenty', 'kickoff_out_of_bounds', 'kickoff_returner_player_id', 'kickoff_returner_player_name', 'lateral_interception_player_id', 'lateral_interception_player_name', 'lateral_kickoff_returner_player_id', 
                        'lateral_kickoff_returner_player_name', 'lateral_punt_returner_player_id', 'lateral_punt_returner_player_name', 'lateral_receiver_player_id', 'lateral_receiver_player_name', 'lateral_reception', 'lateral_recovery', 'lateral_return', 'lateral_rush', 'lateral_rusher_player_id', 'lateral_rusher_player_name', 'lateral_sack_player_id', 'lateral_sack_player_name',
                         'own_kickoff_recovery','own_kickoff_recovery_player_id','own_kickoff_recovery_player_name','own_kickoff_recovery_td','solo_tackle', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name', 'solo_tackle_1_team', 'solo_tackle_2_player_id', 'solo_tackle_2_player_name', 'solo_tackle_2_team','punt_attempt', 'punt_blocked', 'punt_downed', 'punt_fair_catch', 'punt_in_endzone', 
                        'punt_inside_twenty', 'punt_out_of_bounds', 'punt_returner_player_id', 'punt_returner_player_name', 'punter_player_id', 'punter_player_name','total_away_comp_air_epa', 'total_away_comp_air_wpa', 'total_away_comp_yac_epa', 'total_away_comp_yac_wpa', 'total_away_epa', 'total_away_pass_epa', 'total_away_pass_wpa', 'total_away_raw_air_epa', 'total_away_raw_air_wpa', 'total_away_raw_yac_epa', 
                        'total_away_raw_yac_wpa', 'total_away_rush_epa', 'total_away_rush_wpa', 'total_away_score', 'total_home_comp_air_epa', 'total_home_comp_air_wpa', 'total_home_comp_yac_epa', 'total_home_comp_yac_wpa', 'total_home_epa', 'total_home_pass_epa', 'total_home_pass_wpa', 'total_home_raw_air_epa', 'total_home_raw_air_wpa', 'total_home_raw_yac_epa', 'total_home_raw_yac_wpa', 'total_home_rush_epa', 
                        'total_home_rush_wpa', 'total_home_score', 'touchback', 'two_point_attempt', 'two_point_conv_result'])

qb_ids = roster.loc[roster.pos=='QB']
qb_ids = list(qb_ids.player_id.values)
qbs = qbs.loc[(qbs['passer_player_id'].isin(qb_ids)) | qbs['rusher_player_id'].isin(qb_ids)]

qbs['season'] = qbs['season'].astype(str)
qb_rushes = qbs.loc[qbs.play_type=='run']
qb_passes = qbs.loc[qbs.play_type=='pass']

qb_rushes['id'] = qb_rushes['season'] + qb_rushes['posteam'] + qb_rushes['rusher_player_name']
qb_passes['id'] = qb_passes['season'] + qb_passes['posteam'] + qb_passes['passer_player_name']

qb_rushes = pd.merge(how='left',left=qb_rushes,right=roster,on=['id'])
qb_passes = pd.merge(how='left',left=qb_passes,right=roster,on=['id'])

qb_rushes = qb_rushes.drop(columns=['season_y'])
qb_passes = qb_passes.drop(columns=['season_y'])

qb_rushes = qb_rushes.rename(columns={'season_x':'season'})
qb_passes = qb_passes.rename(columns={'season_x':'season'})

# only take QBs that matched roster ids. About 94% of pass plays
qb_passes = qb_passes.loc[qb_passes['player_id']==qb_passes['passer_player_id']]
qb_rushes = qb_rushes.loc[qb_rushes['player_id']==qb_rushes['rusher_player_id']]

qb_passes['sea_id'] = qb_passes['season'].astype(str) + qb_passes['passer_player_id']
qb_rushes['sea_id'] = qb_rushes['season'].astype(str) + qb_rushes['rusher_player_id']

# only take qbs with at least 3 rushes and 100 passes
gb = qb_rushes.groupby(['id'])['epa'].count()
gb = gb.loc[gb >= 3].reset_index()
run_ids = list(gb['id'].unique())

gb = qb_passes.groupby(['id'])['epa'].count()
gb = gb.loc[gb >= 100].reset_index()
pass_ids = list(gb['id'].unique())

# get game count
gc = qb_passes.groupby(['id'])['game_id'].nunique()

# only use qbs with over 5 games
gc = gc.loc[gc > 5]
gc = gc.reset_index()
enough_games = list(gc.id.values)

qb_ids = []
for pid in pass_ids:
    if pid in run_ids:
        if pid in enough_games:
            qb_ids.append(pid)

qb_passes = qb_passes.loc[qb_passes['id'].isin(qb_ids)]
qb_rushes = qb_rushes.loc[qb_rushes['id'].isin(qb_ids)]

gc = gc.loc[gc['id'].isin(qb_ids)]

print(str(len(set(qb_ids))) + " Quarterbacks meet the criteria")

def explosiveness(x):
    # (Warning: O(n**2) in time and memory, where n = len(x).  *Don't* pass in huge samples!)
    
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    mean = np.mean(x)
    
    if mean == 0:
        return 0
    rmad = mad/mean
    # Gini coefficient
    g = 0.5 * rmad
    return g

# brandon weeden has a play in 2013 with a nan value lol which messed shit up
qb_rushes = qb_rushes.dropna(subset=['yards_gained','wpa'])
# yards per rush
ypc = qb_rushes.groupby(['id'])['yards_gained'].mean().reset_index()
ypc = ypc.rename(columns={'yards_gained':'YPC'})
# win probability added on rushes (will do per game later)
wpa = qb_rushes.groupby(['id'])['wpa'].sum().reset_index()
# attempts (will do per game later)
att = qb_rushes.groupby(['id'])['yards_gained'].count().reset_index()
att = att.rename(columns={'yards_gained':'Att'})
# run explosiveness
rexp = qb_rushes.groupby(['id'])['yards_gained'].apply(explosiveness).reset_index()
rexp = rexp.rename(columns={'yards_gained':'Run_Exp'})

# merge in rushing stats
qb_run_dfs = [ypc,wpa,att,rexp]
for rdf in qb_run_dfs:
    gc = pd.merge(how='left',left=gc,right=rdf,on=['id'])
    
gc['Run Att/G'] = gc['Att']/gc['game_id']
gc['Run WPA/G'] = gc['wpa']/gc['game_id']

gc = gc.drop(columns=['Att','wpa'])
# check for more nans
gc1 = gc[gc.isna().any(axis=1)]

def is_success(epa):
    return 1 if epa >= 0 else 0

# add success column
qb_passes['success'] = qb_passes.apply(lambda row: is_success(row['epa']),axis=1)

# total dropbacks
dbs = qb_passes.groupby(['id'])['epa'].count().reset_index()
dbs = dbs.rename(columns={'epa':'dropbacks'})

# average columns to get per dropback numbers
means = qb_passes.groupby(['id'])['touchdown','interception','complete_pass','qb_hit','sack','success','epa','wp'].mean().reset_index()
means = means.rename(columns={'touchdown':'TD/D','interception':'INT/D','complete_pass':'Comp/D','qb_hit':'Hit/D',
                              'sack':'Sack/D','success':'SR/D','epa':'EPA/D','wp':'Avg_WP'})

means['Incomp/D'] = 1-means['TD/D'] - means['INT/D'] - means['Comp/D'] - means['Sack/D']
means.tail(15)

# divide into completions and incompletions and sacks
qb_comps = qb_passes.loc[qb_passes.complete_pass==1]
qb_incomps = qb_passes.loc[(qb_passes.complete_pass==0) & (qb_passes.sack==0)]
qb_sacks = qb_passes.loc[(qb_passes.complete_pass==0) & (qb_passes.sack==1)]

qb_comps = qb_comps.dropna(subset=['yards_after_catch'])
pass_avgs = qb_comps.groupby(['id'])['yards_gained','yards_after_catch','air_yards'].mean().reset_index()
pass_avgs = pass_avgs.rename(columns={'yards_gained':'YPComp','yards_after_catch':'YAC','air_yards':'aDOT'})

# have to calc expl separately
expl_compl = qb_comps.groupby(['id'])['yards_gained'].apply(explosiveness).reset_index()
expl_yac = qb_comps.groupby(['id'])['yards_after_catch'].apply(explosiveness).reset_index()
expl_ay = qb_comps.groupby(['id'])['air_yards'].apply(explosiveness).reset_index()

expl_compl = expl_compl.rename(columns={'yards_gained':'YPComp_Expl'})
expl_yac = expl_yac.rename(columns={'yards_after_catch':'YAC_Expl'})
expl_ay = expl_ay.rename(columns={'air_yards':'AY_Expl'})

icay = qb_incomps.groupby(['id'])['air_yards'].mean().reset_index()
icay = icay.rename(columns={'air_yards':'ICAY'})

# merge in pass dataframes
qb_pass_dfs = [dbs,means,pass_avgs,expl_compl,expl_yac,expl_ay,icay]
for pdf in qb_pass_dfs:
    gc = pd.merge(how='left',left=gc,right=pdf,on=['id'])

gc['Dropbacks/G'] = gc['dropbacks']/gc['game_id']

# check for more nans
gc1 = gc[gc.isna().any(axis=1)]
gc = gc.drop(columns=['game_id','dropbacks'])
# merge in roster
roster_merge = roster[['id','sea_id','age','height','weight','av']]
gc = pd.merge(how='left',left=gc,right=roster_merge,on=['id'])

cols = ['sea_id','age','height','weight','av',
        'Dropbacks/G','EPA/D','SR/D','Comp/D','Incomp/D','TD/D','INT/D','Sack/D','Hit/D','Avg_WP',
        'YPComp','aDOT','YAC','YPComp_Expl','YAC_Expl','AY_Expl','ICAY','YPC','Run_Exp','Run Att/G',
        'Run WPA/G']
gc = gc[cols]

fig, ax = plt.subplots(figsize=(16, 11))

#saleprice correlation matrix
sns.heatmap(gc.corr(),
            cmap='coolwarm',
            annot=True,
            fmt=".2f",
            annot_kws={'size':10},
            cbar=True,
            square=False)

plt.show()

### PCA

from sklearn.preprocessing import StandardScaler

# follows https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# drop nans (32 QBs) - not sure what's wrong, but most are from '09/10 and I'm okay with excluding them
gc = gc.dropna()

# drop duplicates - not sure how these got here
gc = gc.drop_duplicates(subset='sea_id')
features = cols[1:]
labels = cols[:1]
# Separating out the features
x = gc.loc[:, features].values
labels = gc.loc[:,labels]

roster_names = roster[['sea_id','age','season','name']]
roster_names = roster_names.drop_duplicates()
labels = pd.merge(how='left',left=labels, right=roster_names, on=['sea_id'])
labels['label'] = labels['season'].astype(str) + '_' + labels['name']

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
pca_df = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3'])

pca_df = pd.concat([labels,pca_df],axis=1)

# lead_pc1 = ['2011_Aaron Rodgers','2011_Drew Brees']
# lead_pc2 = ['2013_Michael Vick', '2017_Deshaun Watson']
# lead_pc3 = ['2014_Robert Griffin','2017_Drew Brees']

pca_df = pca_df.sort_values(by='pc1',ascending=True).reset_index()

# ar11 = pca_df[pca_df['label']=='2011_Aaron Rodgers'].index.values.astype(int)[0]
# db11 = pca_df[pca_df['label']=='2011_Drew Brees'].index.values.astype(int)[0]
# mv13 = pca_df[pca_df['label']=='2013_Michael Vick'].index.values.astype(int)[0]
# dw17 = pca_df[pca_df['label']=='2017_Deshaun Watson'].index.values.astype(int)[0]
# rg14 = pca_df[pca_df['label']=='2014_Robert Griffin'].index.values.astype(int)[0]
# db17 = pca_df[pca_df['label']=='2017_Drew Brees'].index.values.astype(int)[0]

# print([ar11,db11,mv13,dw17,rg14,db17])

pca_df.head(5)

### Plot in 3d

import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fig = pylab.figure(figsize=(15,9))
ax = fig.add_subplot(111, projection = '3d')

x = pca_df.pc1
y = pca_df.pc2
z = pca_df.pc3

sc = ax.scatter(x,y,z)
# now try to get the display coordinates of the first point

x1, y1, _1 = proj3d.proj_transform(-5.747849,-0.890111,-1.049146, ax.get_proj())
x2, y2, _2 = proj3d.proj_transform(-5.784899, 2.730552,0.447076, ax.get_proj())

label1 = pylab.annotate(
    "Drew Brees 2011", 
    xy = (x1, y1), xytext = (-20, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

label2 = pylab.annotate(
    "Aaron Rodgers 2011", 
    xy = (x2, y2), xytext = (-20, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


def update_position1(e):
    x1, y1, _1 = proj3d.proj_transform(1,1,1, ax.get_proj())
    label1.xy = x1,y1
    label1.update_positions(fig.canvas.renderer)
    
def update_position2(e):
    x2, y2, _2 = proj3d.proj_transform(1,1,1, ax.get_proj())
    label2.xy = x2,y2
    label2.update_positions(fig.canvas.renderer)
    fig.canvas.draw()

fig.canvas.mpl_connect('button_release_event', update_position1)
fig.canvas.mpl_connect('button_release_event', update_position2)

fig.suptitle("3D PCA of QB Seasons 2009-2019", fontsize=20)

pylab.show()

fig.savefig('./plots/QB_3D.png', bbox_inches='tight')



### 2D

x = gc.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pca_df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
pca_df = pd.concat([labels,pca_df],axis=1)

from sklearn.cluster import KMeans,DBSCAN
import random

plt.rcParams.update(plt.rcParamsDefault)
fig, ax = plt.subplots(figsize=(16, 11))

# i like better to be positive
if (pca_df.loc[pca_df['label']=='2011_Aaron Rodgers'].pc1.values[0] <1):
    pca_df['pc1'] = pca_df['pc1'] * -1

X = pca_df[['pc1','pc2']].values

x = pca_df.pc1.values
y = pca_df.pc2.values
n = pca_df.label.values

# clusters = DBSCAN(eps=0.6, min_samples=5).fit_predict(X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
clusters = kmeans.predict(X)
print(clusters.shape)

print(X.shape)
plt.scatter(X[:, 0], X[:, 1], c=clusters, s=50, cmap='viridis')

for i, txt in enumerate(n):
    if ((x[i] >= 4) | (x[i] <=-4)):
        ax.annotate(txt, (x[i], y[i]))
    elif ((y[i] >= 3.5) | (y[i] <= -3.5)):
        ax.annotate(txt, (x[i], y[i]))
    # label some randoms in the middle
    elif ((x[i] <= 4) & (x[i] >=-4) & (y[i] <= 3.5) & (y[i] >= -3.5)):
        if random.random() < 0.15:
            ax.annotate(txt, (x[i], y[i]))
        
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
pca_df['cluster'] = clusters

ax.set_title('2D PCA Viz With  Six Clusters')
ax.set_xlabel('PCA 1: Probably Some Sort of Passing Efficiency Metric')
ax.set_ylabel('PCA 2: Probably Some Sort of Rushing/Explosiveness Metric')

plt.rcParams.update(plt.rcParamsDefault)


fig, ax = plt.subplots(figsize=(16, 11))

# i like better to be positive
if (pca_df.loc[pca_df['label']=='2011_Aaron Rodgers'].pc1.values[0] <1):
    pca_df['pc1'] = pca_df['pc1'] * -1


reference_qbs = ['Josh Allen','Josh Rosen','Lamar Jackson','Carson Wentz','Mitchell Trubisky', 'Patrick Mahomes', 'Baker Mayfield', 'Mitch Trubisky']
jg_df = pca_df.loc[(pca_df['name']=='Jared Goff') | (pca_df['name'].isin(reference_qbs))]

jg_X = jg_df[['pc1','pc2']].values

jg_x = jg_df.pc1.values
jg_y = jg_df.pc2.values
jg_n = jg_df.label.values

# clusters = DBSCAN(eps=0.6, min_samples=5).fit_predict(X)
# kmeans = KMeans(n_clusters=6)
# kmeans.fit(X)
# clusters = kmeans.predict(X)

# print(X.shape)
plt.scatter(jg_X[:, 0], jg_X[:, 1], c=jg_df.cluster, s=50, cmap='viridis')

plt.plot((-6.83289,1.505),(-1.639,0.37868))
plt.plot((1.505,3.253),(0.37868,0.821448))

for i, txt in enumerate(jg_n):
    ax.annotate(txt, (jg_x[i], jg_y[i]))

ax.set_title('Jared Goff ')
ax.set_xlabel('PCA 1: Probably Some Sort of Passing Efficiency Metric')
ax.set_ylabel('PCA 2: Probably Some Sort of Dyanamic Metric')

plt.show()


from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as animation
from matplotlib import colors

def build_first_plot(df):
    
    clusters = list(df.color.values)
    name_list = list(df.label.values)
    x = df['pc1']
    y = df['pc2']

    fig, ax = plt.subplots(figsize=(15,9))
    ax.scatter(x, y, s=50,c=clusters)
    ax.set_title('PC1 vs. PC2',{"fontsize":24})
    plt.title('data from nflscrapR',fontsize=16)
    plt.suptitle('QB PCA Over Time',fontsize=24, y=0.945)
    ax.set_xlabel('PC1',{"fontsize":18})
    ax.set_ylabel('PC2',{"fontsize":18})

    for x0, y0, name in zip(x, y, name_list):
        ax.annotate(name, (x0, y0))

#BUILD THE ANIMATION, THIS WILL SAVE IT TO A MP4 FILE IN YOUR CURRENT DIRECTORY
def build_animation(df):
    x = [[] for i in range(10)]
    for i,j in zip(range(10), range(2009,2019)):
        x[i].append(df[df['season']==j]['pc1'].values.tolist())
        
    y = [[] for i in range(10)]
    for i,j in zip(range(10), range(2009,2019)):
        y[i].append(df[df['season']==j]['pc2'].values.tolist())

    seasons = list(range(2009,2019))

    #DRAW THE FIGURE FIRST
    fig, ax = plt.subplots(figsize=(15,9))
    ax.set_title('PC1 vs. PC2',{"fontsize":24})
    plt.title('data from nflscrapR',fontsize=16)
    plt.suptitle('QB PCA Over Time',fontsize=24, y=0.945)
    ax.set_xlabel('PC1',{"fontsize":18})
    ax.set_ylabel('PC2',{"fontsize":18})

    #ANIMATE OVER THE YEARS
    def animate(i):
        ax.clear()
        plt.xlim(-8,6.5)
        plt.ylim(-4.5,6.5)
        
        season = df.loc[df.season==(2009+i)]
        clusters = list(season.color.values)
        name_list = list(season.label.values)
        
        cmap = colors.ListedColormap(['g','b','y','r','k'])

        x_scatter = x[i][0]
        y_scatter = y[i][0]

        #fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(x_scatter, y_scatter, s=50, c=clusters, cmap=cmap)
        #ax.set_title('Success Rate vs. EPA/Play',{"fontsize":24})
        plt.title('data from nflscrapR',fontsize=16)
        ax.set_xlabel('PC1',{"fontsize":18})
        ax.set_ylabel('PC2',{"fontsize":18})
        ax.text(-6,-2, str(seasons[i]) + '-' + str(seasons[i]+1) + ' Season', fontsize=14)

        for x0, y0, name in zip(x_scatter, y_scatter, name_list):
            ax.annotate(name, (x0, y0))
            
    ani = animation.FuncAnimation(fig, animate, frames=10, interval=5000)
    #plt.show()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Blake'), bitrate=1800)
    ani.save('./plots/animated_scatter.mp4', writer=writer)
    

# some hand edits to the list 
# names18 = list(pca_df.loc[pca_df['season']==2018].name.unique())
# names18.append('Jimmy Garoppolo')
# names18.remove('Jeff Driskel')

desired = ['Ryan Fitzpatrick','Matt Stafford','Alex Smith','Matt Ryan','Joe Flacco','Eli Manning','Ben Roethlisberger','Aaron Rodgers','Tom Brady','Drew Brees']
desired = ['Ryan Fitzpatrick','Eli Manning','Aaron Rodgers','Tom Brady','Drew Brees']

def get_color(x):
    return desired.index(x)

ani_df = pca_df.loc[pca_df['name'].isin(desired)]
ani_df['color'] = ani_df.apply(lambda row: get_color(row['name']),axis=1)
ani_df = ani_df[['label','season','pc1','pc2','color']]
print(ani_df.head())

# # #FIRST LETS MAKE IT JUST FOR 2018
ani_df18 = ani_df[ani_df['season']==2018]

# #builds the standalone 2018 plot
build_first_plot(ani_df18)

# #builds the animation
build_animation(ani_df)