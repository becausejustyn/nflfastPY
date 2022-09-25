

# Making Fantasy Football Projections Via A Monte Carlo Simulation

'''
The first thing we have to do is get the Player objects from nflgame for our team. I wrote a simple function to grab the objects and print them to verify that I have the correct members:
'''

import nflgame
team = ['Drew Brees', 
        'Antonio Brown', 
        'Allen Robinson', 
        'Adrian Peterson',
        'Doug Martin',
         'Gary Barnidge',
         'Keenan Allen']

def make_team(team):
    tm = []
    for p in team:
        for plr in nflgame.find(p):
            if plr.position not in set(['QB','WR','TE','RB']) or plr.status == '':
                continue
            tm.append(plr)
    return tm

def validate_team(team):
    for t in team:
        print(t.full_name, t.team)

tm = make_team(team)
validate_team(tm)

# Scoring nflgame’s output

scoring = {
    'passing_yds' : lambda x : x*.04 +
                        (3. if x >= 300 else 0),
    'passing_tds' : lambda x : x*4., 
    'passing_ints' : lambda x : -1.*x,
    'rushing_yds' : lambda x : x*.1 + (3 if x >= 100 else 0),
    'rushing_tds' : lambda x : x*6.,
    'kickret_tds' : lambda x : x*6.,
    'receiving_tds' : lambda x : x*6.,
    'receiving_yds' : lambda x : x*.1,
    'receiving_rec' : lambda x : x,
    'fumbles_lost' : lambda x : -1*x,
    'passing_twoptm'  : lambda x : 2*x,
    'rushing_twoptm' : lambda x : 2*x,
    'receiving_twoptm' : lambda x : 2*x
}

def score_player(player):
    score = 0
    for stat in player._stats:
        if stat in scoring:
            score += scoring[stat](getattr(player,stat))    
    return score

# Simulating the score for a single player


2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
import numpy as np
def get_score_for_player(player):
    
    # Sample the year and week
    year = np.random.choice([2013,2014,2015],
                            p=[.2,.3,.5])
    week = np.random.randint(1,18)
    
    # Find the player and score them for the given week/year   
    for p in get_games(year,week):
        if p.player is None:
            continue
        if player == p.player:
            return score_player(p)
        
    return get_score_for_player(player) # Retry due to bye weeks / failure for any other reason


#Defining the get_games function and using the LRU Cache decorator for performance

'''
The get_game function is a wrapper for nflgame which I define below. 
It can be a costly function because nflgame stores data in zipped files on disk (if it is not pinging the NFL servers).

The get_game function called is defined below. 
I use the lru_cache decorator to set up a cache for games returned so the code won’t have to ping nflgame for the data if it’s already been accessed before. 
This is a simple approach to more efficiently dealing with a library which may have costly function or data access calls.

The inputs and the output of the function must be hashable for this to work Under the covers, the lru_cache will create a dict which stores inputs to outputs. 
If you call the function with the same inputs as a previous call in the cache, it will automatically return to you the output without actually calling the function.
'''

from functools import lru_cache
@lru_cache(200) # Define a cache with 200 empty slots
def get_games(year,week):
    g = nflgame.games(year,week=week)
    return nflgame.combine_game_stats(g)

# Simulation and Results
import pandas as pd
def simulate(team, exps=10):
    scores = pd.DataFrame(data=np.zeros((exps,len(team))),
                          columns = [p.name for p in team])
    for n in range(exps):
        for player in team:
            scores.loc[n,player.name] += get_score_for_player(player)
    return scores


outcome = simulate(tm, exps=100)
outcome.head()

# Projecting Fantasy Points
game_points = outcome.sum(axis=1, skipna=True) # Sum the player scores together

print('Team projection: %s' % game_points.mean())
print('Standard Deviations: %s' % (game_points.std()/np.sqrt(len(outcome.columns))))


## player level stats

outcome.mean() # Point projections for each player
outcome.std()  # Standard deviation in point projections for each player
