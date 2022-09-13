

###############################   DICT   #################################

REDUCED = {
        "Arizona": "ARI", "ARZ": "ARI",
        "Atlanta": "ATL",
        "Baltimore": "BAL",
        "Buffalo": "BUF",
        "Carolina": "CAR",
        "Chicago": "CHI",
        "Cincinnati": "CIN",
        "Cleveland": "CLE",
        "Dallas": "DAL",
        "Denver": "DEN",
        "Detroit": "DET",
        "Green Bay": "GB",
        "Houston": "HOU",
        "Indianapolis": "IND",
        "Jacksonville": "JAX",
        "Kansas City": "KC",
        "Los Angeles": "LA",
        "Miami": "MIA",
        "Minnesota": "MIN",
        "New England": "NE",
        "New Orleans": "NO",
        "NY Giants": "NYG",
        "NY Jets": "NYJ",
        "Oakland": "OAK",
        "Philadelphia": "PHI",
        "Pittsburgh": "PIT",
        "San Diego": "SD",
        "Seattle": "SEA",
        "San Francisco": "SF",
        "Tampa Bay": "TB",
        "Tennessee": "TEN",
        "Washington": "WSH"
}

ROW_NUM = {
        "ARI": 0,
        "ATL": 1,
        "BAL": 2,
        "BUF": 3,
        "CAR": 4,
        "CHI": 5,
        "CIN": 6,
        "CLE": 7,
        "DAL": 8,
        "DEN": 9,
        "DET": 10,
        "GB": 11,
        "HOU": 12,
        "IND": 13,
        "JAX": 14,
        "KC": 15,
        "LA": 16,
        "MIA": 17,
        "MIN": 18,
        "NE": 19,
        "NO": 20,
        "NYG": 21,
        "NYJ": 22,
        "OAK": 23,
        "PHI": 24,
        "PIT": 25,
        "SD": 26,
        "SEA": 27,
        "SF": 28,
        "TB": 29,
        "TEN": 30,
        "WSH": 31,
}

team_abv = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Rams": "LAR",
    "Chargers": "LAC",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Washington": "WSH",
    "Bye": "BYE",
}

city_team = {
    "Vegas": "Raiders",
    "Jets": "Jets",
    "Indianapolis": "Colts",
    "Houston": "Texans",
    "Cleveland": "Browns",
    "Tennessee": "Titans",
    "Detroit": "Lions",
    "Chicago": "Bears",
    "Jacksonville": "Jaguars",
    "Orleans": "Saints",
    "Atlanta": "Falcons",
    "Angeles": "Rams",
    "Arizona": "Cardinals",
    "Giants": "Giants",
    "Seattle": "Seahawks",
    "England": "Patriots",
    "Philadelphia": "Eagles",
    "Bay": "Packers",
    "Denver": "Broncos",
    "City": "Chiefs",
    "Washington": "Washington",
    "Pittsburgh": "Steelers",
    "Buffalo": "Bills",
    "Francisco": "49ers",
    "Baltimore": "Ravens",
    "Dallas": "Cowboys",
    "Carolina": "Panthers",
    "Tampa": "Buccaneers",
    "Cincinatti": "Bengals",
    "Miami": "Dolphins",
}

divisions_dict = {
    "NFC_East": ["Eagles", "Giants", "Cowboys", "Washington"],
    "NFC North": ["Bears", "Packers", "Lions", "Vikings"],
    "NFC South": ["Buccaneers", "Saints", "Panthers", "Falcons"],
    "NFC West": ["Seahawks", "Cardinals", "Rams", "49ers"],
    "AFC East": ["Bills", "Dolphins", "Patriots", "Jets"],
    "AFC North": ["Steelers", "Ravens", "Browns", "Bengals"],
    "AFC South": ["Titans", "Colts", "Texans", "Jaguars"],
    "AFC West": ["Chiefs", "Raiders", "Broncos", "Chargers"],
}

records = {
    "Cardinals": [0, 0, 0],
    "Falcons": [0, 0, 0],
    "Ravens": [0, 0, 0],
    "Bills": [0, 0, 0],
    "Panthers": [0, 0, 0],
    "Bears": [0, 0, 0],
    "Bengals": [0, 0, 0],
    "Browns": [0, 0, 0],
    "Cowboys": [0, 0, 0],
    "Broncos": [0, 0, 0],
    "Lions": [0, 0, 0],
    "Packers": [0, 0, 0],
    "Texans": [0, 0, 0],
    "Colts": [0, 0, 0],
    "Jaguars": [0, 0, 0],
    "Chiefs": [0, 0, 0],
    "Raiders": [0, 0, 0],
    "Rams": [0, 0, 0],
    "Chargers": [0, 0, 0],
    "Dolphins": [0, 0, 0],
    "Vikings": [0, 0, 0],
    "Patriots": [0, 0, 0],
    "Saints": [0, 0, 0],
    "Giants": [0, 0, 0],
    "Jets": [0, 0, 0],
    "Eagles": [0, 0, 0],
    "Steelers": [0, 0, 0],
    "49ers": [0, 0, 0],
    "Seahawks": [0, 0, 0],
    "Buccaneers": [0, 0, 0],
    "Titans": [0, 0, 0],
    "Washington": [0, 0, 0],
}



# https://blog.collegefootballdata.com/talking-tech-elo-ratings/

import datetime
import numpy as np
import pandas as pd

def calc_expected_score(team_rating, opp_team_rating):
    return 1 / (1 + 10**((opp_team_rating - team_rating) / 400))

def calc_new_rating(team_rating, observed_score, expected_score, k_factor = 20):
    return team_rating + k_factor * (observed_score - expected_score)

def get_expected_score(rating, opp_rating):
    exp = (opp_rating - rating) / 400
    return 1 / (1 + 10**exp)

print(f'{get_expected_score(1500, 1500):.3f}')
print(f'{get_expected_score(1400, 1500):.3f}')
print(f'{get_expected_score(2000, 1500):.3f}')

def get_new_elos(home_rating, away_rating, margin, k = 25):
    # 0.5 for tie, 1 for win, 0 for loss
    home_score = np.where(margin > 0, 1, np.where(margin < 0, 0, 0.5))
  
    # get expected home score
    expected_home_score = get_expected_score(home_rating, away_rating)
    # multiply difference of actual and expected score by k value and adjust home rating
    new_home_score = home_rating + k * (home_score - expected_home_score)

    # repeat these steps for the away team
    # away score is inverse of home score
    away_score = 1 - home_score
    expected_away_score = get_expected_score(away_rating, home_rating)
    new_away_score = away_rating + k * (away_score - expected_away_score)

    # return a tuple
    return(new_home_score, new_away_score)

get_new_elos(1750, 1500, 7)



# initial_ratings = pd.DataFrame({'team': df1['home_team'].unique(), 'elo': 1500, 'week': 0})

### Misc Functions

def date_sort(game):
    game_date = datetime.datetime.strptime(game['start_date'], "%Y-%m-%dT%H:%M:%S.000Z")
    return game_date

def elo_sort(team):
    return team['elo']

def get_row_num(name):
    try:
        try:
            name = REDUCED[name]
        except KeyError:
            pass
        row_num = ROW_NUM[name]
        return row_num
    except KeyError as err:
        print("Can't find team name {}".format(name))
        return False

def get_row_scores(score_dict, week):
    row_scores = [""] * 32
    for game in score_dict['page']:
        try:
            home = game['homeTeam']['abbreviation']
            away = game['awayTeam']['abbreviation']
            home_score = game['score']['homeScore']
            away_score = game['score']['awayScore']
            row_scores[get_row_num(home)] = home_score - away_score
            row_scores[get_row_num(away)] = away_score - home_score
        except KeyError as err:
            row_scores[get_row_num(home)] = ""
            row_scores[get_row_num(away)] = ""
    return row_scores