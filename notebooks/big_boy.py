
import os
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

#FIELD_GOAL_RANGE = 66 # In simulations, drives dying before this point will punt (regardless of game situation).
#PIECES = [0, 15, 40, 66, 99] # Divide the field up into chunks for piecewise exponential modeling.
#REDZONE_PIECE = 3
#PIECES = [0, 13, 66, 99]
#REDZONE_PIECE = 2

FIELD_GOAL_RANGE = 67
PIECES = [0, 13, 75, 99]
REDZONE_PIECE = 2
MEDIAN_I = 32

PUNT_DISTANCE = 38 # net of return
PUNT_CLOCK_TIME = 15./60
TOUCHDOWN_CLOCK_TIME = 15./60
FG_CLOCK_TIME = 15./60
LOSING_BADLY_THRESHOLD = 16
SIMULATION_TIE_BREAKER_COIN_FLIP_HOME_ADVANTAGE = .03

MINUTES_PER_YARD = 0
MINUTES_INTERCEPT = 0
READY_TO_USE = False

def piece_lengths(PIECES):
    return [PIECES[i + 1] - PIECES[i] for i in range(len(PIECES) - 1)]

PIECE_LENGTHS = piece_lengths(PIECES)

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(os.getcwd(), 'data/')
CHART_DIR = os.path.join(os.getcwd(), 'charts/')
GAME_LEVEL_DATASET_2014 = DATA_DIR + 'nfl_game_level.xlsx'
DOME_DATASET = DATA_DIR + 'nfl_home_stadiums.csv'

### handy lists of columns for looking at the data.
DRIVE_COLS = ['GameId', 'quarter', 'minute', 'second', 'drive_id', 'OffenseTeam', u'DefenseTeam', 'start_yardline',
              'first_downs', 'yards_zero_plus', 'yards_end_minus_start_zero_plus', 'end_yardline_zero_plus',
              'end_turnover', 'end_touchdown', 'end_field_goal_attempt']
PLAY_COLS = ['GameId', 'drive_id', 'play_id', 'Quarter', 'Minute', 'Second', 'OffenseTeam', u'DefenseTeam', 'Yards',
             'is_offensive_touchdown', 'is_field_goal', 'Down', 'ToGo']

def enrich_play_level_df(df):
    '''
    Add helpful columns - mostly boolean - to the drive level dataset, identifying play types and outcomes.
    Generate unique drive_ids grouping plays into drives.

    :param df: drive level data set from http://nflsavant.com/about.php
    :return: df
    '''
    df = df.sort_index(by=['GameId', 'Quarter', 'Minute', 'Second'], ascending=[True, True, False, False])
    df = df.reset_index().drop('index', axis=1)

    df['clock'] = 60 - (df.Quarter) * 15 + df.Minute + df.Second / 60.0
    df['clock_after_play'] = df.clock.shift(-1)

    # drop plays we obviously dont need
    # df = df[(df.PlayType != 'TIMEOUT')]
    df = df[(df.Description.notnull())]
    df = df[~(df.Description.str.contains('TWO-MINUTE WARNING'))]
    df = df[~(df.Description.str.contains('TIMEOUT')) |
            (df.Description.str.len() > 35)]  # some timeouts tacked onto play descriptions

    # Tag final play of half, so we'll know if a drive was censored by the clock
    df['is_end_of_quarter'] = (df.Description.str.contains(r'END.+QUARTER'))
    df['is_end_of_period'] = (df.Description.str.contains('END GAME')) | \
                             (df.Description.str.contains('END OF GAME')) | \
                             (df.Description.str.contains('END.+HALF')) | \
                             (df.Description.str.contains(r'END.+QUARTER')) & \
                             (df.Quarter.isin([2, 4]))
    df['is_final_play_of_half'] = df.is_end_of_period.shift(-1).fillna(False)

    # Some plays lack the OffenseTeam variable.  Interpolate it.
    df['Offense_ffill'] = df.OffenseTeam.fillna(method='ffill')
    df.OffenseTeam = df.OffenseTeam.fillna(method='backfill')
    df.loc[df.is_end_of_period, 'OffenseTeam'] = df.Offense_ffill  # don't backfill from other games!

    # Generate indicators we care about, using flags in dataset, PlayType column, and parsing of Description col.
    df.IsFumble = df.IsFumble.astype(bool)
    df.IsInterception = df.IsInterception.astype(bool)
    df['is_offensive_touchdown'] = (df.IsTouchdown == 1) & ~(df.IsFumble | df.IsInterception) & \
                                   (df.IsChallengeReversed != 1)
    df['is_punt'] = (df.PlayType == 'PUNT')
    df['is_kick_off'] = (df.PlayType == 'KICK OFF')
    df['is_field_goal'] = (df.PlayType == 'FIELD GOAL')
    df['is_field_goal_not_nullified'] = (df.PlayType == 'FIELD GOAL') & (df.IsPenaltyAccepted != 1)
    df['is_extra_point'] = (df.PlayType == 'EXTRA POINT')
    df['is_extra_point_successful'] = (df.is_extra_point) & (df.Description.str.contains('IS GOOD'))
    df['is_after_touchdown'] = (df.PlayType.isin(['EXTRA POINT', 'TWO-POINT CONVERSION']))
    df['is_time_out'] = (df.PlayType == 'TIMEOUT')
    df['is_no_play'] = (df.Description.str.contains('NO PLAY'))
    df['is_qb_kneel'] = (df.PlayType == 'QB KNEEL')
    df['is_missed_field_goal'] = (df.is_field_goal) & \
                                 (df.Description.str.contains('NO GOOD') | df.Description.str.contains('BLOCKED'))
    df['is_safety'] = (df.Description.str.contains('SAFETY'))
    df['is_turnover'] = (df.IsFumble | df.IsInterception) & (df.OffenseTeam != df.OffenseTeam.shift(-1))
    df['is_turnover_on_missed_fg'] = (df.is_missed_field_goal) & (df.OffenseTeam != df.OffenseTeam.shift(-1))
    df['is_turnover_on_downs'] = (df.Down == 4) & \
                                 (df.IsTouchdown == 0) & \
                                 (~df.is_field_goal) & \
                                 ~(df.is_punt) & \
                                 ~(df.is_turnover) & \
                                 (df.OffenseTeam != df.OffenseTeam.shift(-1))

    # penalties
    df['is_offensive_penalty'] = (df.IsPenaltyAccepted == 1) & (df.PenaltyTeam == df.OffenseTeam)
    df['is_defense_penalty'] = (df.IsPenaltyAccepted == 1) & (df.PenaltyTeam == df.DefenseTeam)

    # Break up plays into drives.
    df['new_drive'] = df.is_turnover.shift(1) | df.is_kick_off.shift(1) | \
                      df.is_punt.shift(1) | df.is_turnover_on_downs.shift(1) | df.is_turnover_on_missed_fg.shift(1)
    df.new_drive = df.new_drive.fillna(False)
    df['drive_id'] = df.new_drive.astype(int).cumsum()

    # Un-tag non-drive plays
    df['is_drive_play'] = ~(df.is_kick_off) & \
                          ~(df.is_end_of_quarter) & \
                          ~(df.is_after_touchdown) & \
                          ~(df.is_end_of_period)
    # ~(df.PlayType.isin(['NO PLAY']))
    df.loc[~df.is_drive_play, 'drive_id'] = None

    df['is_first_play_of_drive'] = (df.drive_id != df.drive_id.shift(1)) & (df.drive_id.notnull())
    df['is_last_play_of_drive'] = (df.drive_id != df.drive_id.shift(-1)) & (df.drive_id.notnull())
    df['yards_offensive'] = df.apply(offensive_yards, axis=1)
    df['yardline_after_play'] = df.apply(
        lambda x: x['YardLine'] if x['is_no_play'] else x['YardLine'] + x['yards_offensive'],
        axis=1)
    df['is_earned_first_down'] = (df.Down == 1) & \
                                 (df.yards_offensive.shift(1) > df.ToGo.shift(1)) & \
                                 (df.OffenseTeam.shift(1) == df.OffenseTeam)
    df['play_id'] = df.index
    return df


def merge_in_game_level_dataset(df, game_level_dataset):
    """
    Merge in home and away team data
    :param df:
    :param game_level_dataset:
    :return:
    """
    df_game = load_game_level_dataset(game_level_dataset)
    df = pd.merge(df, df_game, on='GameId', how='left')
    return df


def load_game_level_dataset(game_level_dataset):
    df_game = pd.read_excel(game_level_dataset)
    df_game = df_game[['GameId', 'hometeam', 'awayteam']]
    return df_game


def merge_team_indexes_with_game_level_df(df_game, teams):
    df_game = pd.merge(df_game, teams, left_on='hometeam', right_on='slug')
    df_game = df_game.rename(columns={'i': 'i_home'}).drop('slug', axis=1)
    df_game = pd.merge(df_game, teams, left_on='awayteam', right_on='slug')
    df_game = df_game.rename(columns={'i': 'i_away'}).drop('slug', axis=1)
    return df_game


def calculate_game_score_at_play_start(df):
    """
    :param df:
    :param game_level_dataset:
    :return:
    """
    points_on_play = pd.DataFrame(df.apply(points_scored, axis=1))
    df = df.join(points_on_play)

    df['home_points'] = (df.hometeam == df.OffenseTeam).astype(int) * df.points_o
    df['home_points'] += (df.hometeam == df.DefenseTeam).astype(int) * df.points_d
    df['away_points'] = (df.awayteam == df.OffenseTeam).astype(int) * df.points_o
    df['away_points'] += (df.awayteam == df.DefenseTeam).astype(int) * df.points_d

    g = df.groupby('GameId')

    df['home_score'] = g.home_points.cumsum()
    df['away_score'] = g.away_points.cumsum()

    return df


def generate_drive_df(df):
    """
    Generate a frame with drive-level stats and metadata.
    Drops a handful of drives that (due to errors in dataset and edge cases) have non-unique (offense, defense) teams
    :param df:
    :return: drive_df
    """
    g = df.groupby('drive_id')
    df_drive = pd.DataFrame({'end_touchdown': g.is_offensive_touchdown.max(),
                             'end_field_goal_attempt': g.is_field_goal_not_nullified.max(),
                             'end_turnover': g.is_turnover.max(),
                             'end_turnover_on_downs': g.is_turnover_on_downs.max(),
                             'end_turnover_missed_fg': g.is_turnover_on_missed_fg.max(),
                             'end_due_to_clock': g.is_final_play_of_half.max(),
                             'end_punt': g.is_punt.max(),
                             'end_safety': g.is_safety.max(),
                             'end_qb_kneel': g.is_qb_kneel.max(),
                             'yards': g.yards_offensive.sum(),
                             'first_downs': g.SeriesFirstDown.sum(),
                             'first_downs_earned': g.is_earned_first_down.sum(),
                             'first_play_of_drive': g.play_id.min(),
                             'last_play_of_drive': g.play_id.max()})
    df_drive['drive_id'] = df_drive.index

    # yardline, clock, score at drive start...
    drive_start = df[['Quarter', 'Minute', 'Second', 'play_id', 'YardLine', 'clock', 'home_score', 'away_score']]
    drive_start.columns = ['quarter', 'minute', 'second', 'play_id', 'start_yardline', 'start_clocktime',
                           'start_home_score', 'start_away_score']
    df_drive = pd.merge(df_drive, drive_start, left_on='first_play_of_drive', right_on='play_id', how='left')
    df_drive = df_drive.drop('play_id', axis=1)

    # ... and drive end.
    drive_end = df[['play_id', 'yardline_after_play', 'clock_after_play']]
    drive_end.columns = ['play_id', 'end_yardline', 'end_clocktime']
    df_drive = pd.merge(df_drive, drive_end, left_on='last_play_of_drive', right_on='play_id', how='left')
    df_drive = df_drive.drop('play_id', axis=1)
    df_drive['yards_end_minus_start'] = df_drive.end_yardline - df_drive.start_yardline
    df_drive['yards_end_minus_start_zero_plus'] = df_drive.yards_end_minus_start.apply(lambda x: max(0, x))
    df_drive['end_yardline_zero_plus'] = df_drive.apply(lambda x: max(x['start_yardline'], x['end_yardline']), axis=1)
    df_drive['elapsed_clock'] = df_drive.start_clocktime - df_drive.end_clocktime
    df_drive.loc[df_drive.elapsed_clock < -50, 'elapsed_clock'] = df_drive.start_clocktime

    #
    drive_teams_and_game = df[df.drive_id.notnull() &
                              (df.OffenseTeam != df.DefenseTeam) &
                              df.OffenseTeam.notnull() &
                              df.DefenseTeam.notnull()].drop_duplicates(
        subset=['GameId', 'drive_id', 'OffenseTeam', 'DefenseTeam'])
    drive_teams_and_game = drive_teams_and_game[
        ['GameId', 'drive_id', 'OffenseTeam', 'DefenseTeam', 'hometeam', 'awayteam']]

    dupes = drive_teams_and_game.drive_id.value_counts()
    dupes = dupes[dupes > 1].index.values

    print('Dropping %s drives due to non-unique offense/defense teams.' % len(dupes))
    drive_teams_and_game = drive_teams_and_game[~drive_teams_and_game.drive_id.isin(dupes)]

    df_drive = pd.merge(drive_teams_and_game, df_drive, on='drive_id', how='left')
    return df_drive


def merge_stadium_dataset(df_drive):
    """
    :param df_game:
    :return:
    """
    df_s = pd.read_csv(DOME_DATASET)[['slug', 'has_dome']]
    df_s.columns = ['hometeam', 'has_dome']
    df_s.has_dome = df_s.has_dome.astype(bool)
    df_drive = pd.merge(df_drive, df_s, on='hometeam', how='left')
    return df_drive


def remove_unexplained_drives(df_drive):
    """
    Remove the handful of drives having no known outcome.  Caused by errors in dataset and edge cases.
    :param df_drive:
    :return: df_drive
    """
    df_drive['explained'] = df_drive.end_due_to_clock | df_drive.end_field_goal_attempt | \
                            df_drive.end_punt | df_drive.end_touchdown | \
                            df_drive.end_turnover | df_drive.end_turnover_missed_fg | \
                            df_drive.end_turnover_on_downs | df_drive.end_safety

    print('dropping %s unexplained drives.' % df_drive[~df_drive.explained].shape[0])
    return df_drive[df_drive.explained]


def index_with_team_indexes(df_drive):
    """
    Assign each team an integer id number, and determine ids of home/away/attacking/defending teams
    :param df_drive:
    :return: df_drive
    """
    # alpha-sorted team slugs ('ARI', 'ATL', 'BAL'...)
    teams = pd.DataFrame({'slug': df_drive.OffenseTeam.unique()}).sort_index(by=['slug']).reset_index().drop('index', 1)
    teams['i'] = teams.index
    df_drive = pd.merge(df_drive, teams, left_on='hometeam', right_on='slug')
    df_drive = df_drive.rename(columns={'i': 'i_home'}).drop('slug', axis=1)
    df_drive = pd.merge(df_drive, teams, left_on='awayteam', right_on='slug')
    df_drive = df_drive.rename(columns={'i': 'i_away'}).drop('slug', axis=1)
    df_drive = pd.merge(df_drive, teams, left_on='OffenseTeam', right_on='slug')
    df_drive = df_drive.rename(columns={'i': 'i_attacking'}).drop('slug', axis=1)
    df_drive = pd.merge(df_drive, teams, left_on='DefenseTeam', right_on='slug')
    df_drive = df_drive.rename(columns={'i': 'i_defending'}).drop('slug', axis=1)
    return df_drive, teams


def offensive_yards(x):
    """
    Yards - including penalty yards - earned on a play.  Turnovers count as 0.
    :param x: row in play data frame
    :return: int
    """

    if x['PlayType'] == 'PUNT' or x['is_turnover']:
        return 0
    if x['IsPenaltyAccepted'] and x['PenaltyTeam'] == x['DefenseTeam']:
        return x['PenaltyYards'] + x['Yards']
    if x['IsPenaltyAccepted'] and x['PenaltyTeam'] == x['OffenseTeam']:
        if x['IsTouchdown']:
            return x['Yards']  # usually personal foul for excessive celebration, not relevant to drive
        # if 'FACE MASK' in x['Description']:
        return x['Yards'] - x['PenaltyYards']
        # return -1 * x['PenaltyYards']
    if x['is_no_play']:
        return 0
    return x['Yards']


def points_scored(x):
    """
    Calculate points scored by offense/defense on a given play.
    Known to be inaccurate in edge cases.

    :param x: row in play data frame
    :return: (offensive points, defensive points)
    """
    if x['IsTouchdown']:
        if x['is_turnover']:
            return pd.Series({'points_o': 0, 'points_d': 6})
        return pd.Series({'points_o': 6, 'points_d': 0})
    if x['is_extra_point_successful']:
        return pd.Series({'points_o': 1, 'points_d': 0})
    if x['IsTwoPointConversionSuccessful']:
        return pd.Series({'points_o': 2, 'points_d': 0})
    if x['is_field_goal'] and not x['is_missed_field_goal']:
        return pd.Series({'points_o': 3, 'points_d': 0})
    if x['is_safety']:
        return pd.Series({'points_o': 0, 'points_d': 2})
    return pd.Series({'points_o': 0, 'points_d': 0})


def enrich_drive_level_df(df_drive, losing_badly_threshold=LOSING_BADLY_THRESHOLD):
    """
    :param df_drive:
    :param losing_badly_threshold:
    :return: df_drive
    """
    df_drive['yards_zero_plus'] = df_drive.yards.apply(lambda x: max(0, x))
    df_drive['is_failure'] = (~df_drive.end_touchdown & ~df_drive.end_due_to_clock).astype(int)
    df_drive['is_censored'] = (df_drive.end_touchdown | df_drive.end_due_to_clock).astype(int)

    df_drive['is_failure_turnover'] = (df_drive.end_turnover).astype(int)
    df_drive['is_censored_turnover'] = (~(df_drive.end_turnover)).astype(int)

    df_drive['defending_team_is_home'] = (df_drive.DefenseTeam == df_drive.hometeam).astype(int)
    df_drive['offensive_team_is_home'] = (df_drive.OffenseTeam == df_drive.hometeam).astype(int)
    df_drive['two_minute_drill'] = ((df_drive.start_clocktime < 32) & (df_drive.start_clocktime > 30)) | \
                                   (df_drive.start_clocktime < 2)
    df_drive['thirty_seconds_drill'] = ((df_drive.start_clocktime < 30.5) & (df_drive.start_clocktime > 30)) | \
                                       (df_drive.start_clocktime < .5)
    df_drive['start_offense_score'] = (df_drive.hometeam == df_drive.OffenseTeam).astype(
        int) * df_drive.start_home_score
    df_drive['start_offense_score'] += (df_drive.awayteam == df_drive.OffenseTeam).astype(
        int) * df_drive.start_away_score
    df_drive['start_defense_score'] = (df_drive.hometeam == df_drive.DefenseTeam).astype(
        int) * df_drive.start_home_score
    df_drive['start_defense_score'] += (df_drive.awayteam == df_drive.DefenseTeam).astype(
        int) * df_drive.start_away_score
    df_drive['offense_losing_badly'] = (
                                           df_drive.start_defense_score - df_drive.start_offense_score) > losing_badly_threshold
    df_drive['offense_winning_greatly'] = (
                                              df_drive.start_offense_score - df_drive.start_defense_score) > losing_badly_threshold
    return df_drive


def generate_piecewise_df(df_drive):
    """
    Expand a drive level frame into a piecewise frame with all drives passing through or dying in that piece.
    :param df_drive:
    :return: df_pw
    """
    new_frames = []
    for i in range(len(PIECES) - 1):
        lower, upper = PIECES[i], PIECES[i + 1]
        df_piece = df_drive[(df_drive.start_yardline < upper) & (df_drive.end_yardline_zero_plus > lower)]
        df_piece['died_in_piece'] = (df_piece.end_yardline_zero_plus <= upper)
        df_piece['died_in_piece_ex_turnover'] = (df_piece.end_yardline_zero_plus <= upper) & ~(df_piece.end_turnover)
        df_piece['died_in_piece_turnover'] = (df_piece.end_yardline_zero_plus <= upper) & (df_piece.end_turnover)
        df_piece['exposure_start'] = df_piece.apply(lambda x: max(x['start_yardline'], lower), axis=1)
        df_piece['exposure_end'] = df_piece.apply(lambda x: min(x['end_yardline_zero_plus'], upper), axis=1)
        df_piece['exposure_yards'] = df_piece.exposure_end - df_piece.exposure_start
        df_piece['exposure_yards'] = df_piece.apply(lambda x: max(.01, x['exposure_yards']), axis=1)
        df_piece['piece_i'] = i
        df_piece['piece_lower'] = lower
        df_piece['piece_upper'] = upper
        new_frames.append(df_piece.copy())
    df_pw = pd.concat(new_frames, ignore_index=True)
    return df_pw


def generate_piecewise_counts_df(df_pw):
    g = df_pw.groupby(['piece_i',
                       'i_attacking', 'i_defending',
                       'i_home', 'i_away',
                       'defending_team_is_home',
                       'offense_losing_badly',
                       'offense_winning_greatly',
                       'two_minute_drill'])
    df = pd.DataFrame({'exposure_yards': g.exposure_yards.sum(),
                       'deaths': g.died_in_piece.sum(),
                       'deaths_turnover': g.died_in_piece_turnover.sum(),
                       'deaths_ex_turnover': g.died_in_piece_ex_turnover.sum(),
                       'N': g.size()
    })
    df = df.reset_index()
    for c in ['i_attacking', 'i_defending', 'i_home', 'i_away']:
        df[c] = df[c].astype(int)
    return df

def set_elapsed_time_per_yard(df_drive):
    y, X = dmatrices('elapsed_clock ~ yards_end_minus_start_zero_plus', data=df_drive, return_type='dataframe')
    model = sm.OLS(y, X)
    results = model.fit()
    global MINUTES_PER_YARD
    MINUTES_PER_YARD = results.params['yards_end_minus_start_zero_plus']
    global MINUTES_INTERCEPT
    MINUTES_INTERCEPT = results.params['Intercept']
    global READY_TO_USE
    READY_TO_USE = True
    return


def drive_time_elapsed(drive_length):
    if READY_TO_USE:
        return MINUTES_INTERCEPT + MINUTES_PER_YARD * drive_length
    else:
        print('Run set_elapsed_time_per_yard() first to fit model.')
        raise Exception


class ParamCalculator(object):
    """
    Some ex-turnover models break out the redzone, others don't.
    Some turnover models take into account defense takeaway propensity, others don't.
    The idea is to have one simulation function, and have these classes
    do the work of providing the appropriate parameters.
    """

    def __init__(self, ex_turnover, turnover):
        self.ex_turnover = ex_turnover
        self.turnover = turnover
        self.ex_t = {}
        self.t = {}
        self.i_home = None
        self.i_away = None

    def re_draw_sample(self):
        self.ex_t = self.re_draw_ex_turnover_sample()
        self.t = self.re_draw_turnover_sample()
        self.tack_on_median_team()

    def re_draw_ex_turnover_sample(self):
        num_samples = self.ex_turnover.atts.gettrace().shape[0]
        draw = np.random.randint(0, num_samples)
        return {'atts': self.ex_turnover.atts.gettrace()[draw, :],
                'atts_rz': self.ex_turnover.atts_rz.gettrace()[draw, :],
                'defs': self.ex_turnover.defs.gettrace()[draw, :],
                'defs_rz': self.ex_turnover.defs_rz.gettrace()[draw, :],
                'home': self.ex_turnover.home.gettrace()[draw, :],
                'baseline_hazards': self.ex_turnover.baseline_hazards.gettrace()[draw, :],
                'two_minute_drill': self.ex_turnover.two_minute_drill.gettrace()[draw],
                'offense_losing_badly': self.ex_turnover.offense_losing_badly.gettrace()[draw],
                'offense_winning_greatly': self.ex_turnover.offense_winning_greatly.gettrace()[draw]}

    def re_draw_turnover_sample(self):
        num_samples = self.turnover.atts.gettrace().shape[0]
        draw = np.random.randint(0, num_samples)
        return {'atts': self.turnover.atts.gettrace()[draw, :],
                'defs': self.turnover.defs.gettrace()[draw, :],
                'home': self.turnover.home.gettrace()[draw],
                'baseline_hazards': self.turnover.baseline_hazards.gettrace()[draw, :],
                'two_minute_drill': self.turnover.two_minute_drill.gettrace()[draw],
                'offense_losing_badly': self.turnover.offense_losing_badly.gettrace()[draw],
                'offense_winning_greatly': self.turnover.offense_winning_greatly.gettrace()[draw]}


    def tack_on_median_team(self):
        """
        Create a 33rd "team" representing the median team.  We'll use this median team to assess team quality and
        strength of schedule.
        :param ex_turnover_params:
        :param turnover_params:
        :return:
        """
        for team_specific_param in ['atts', 'atts_rz', 'defs', 'defs_rz', 'home']:
            if team_specific_param in self.ex_t:
                self.ex_t[team_specific_param] = np.append(self.ex_t[team_specific_param],
                                                           np.median(self.ex_t[team_specific_param]))

        for team_specific_param in ['atts', 'defs']:
            if team_specific_param in self.t:
                self.t[team_specific_param] = np.append(self.t[team_specific_param],
                                                        np.median(self.t[team_specific_param]))


    def home_xb(self):
        return self.ex_t['atts'][self.i_home] + self.ex_t['defs'][self.i_away]

    def home_xb_rz(self):
        return self.ex_t['atts_rz'][self.i_home] + self.ex_t['defs_rz'][self.i_away]

    def away_xb(self):
        return self.ex_t['atts'][self.i_away] + self.ex_t['defs'][self.i_home] + self.ex_t['home'][self.i_home]

    def away_xb_rz(self):
        return self.ex_t['atts_rz'][self.i_away] + self.ex_t['defs_rz'][self.i_home] + self.ex_t['home'][self.i_home]

    def home_turnover_xb(self):
        return self.t['atts'][self.i_home] + self.t['defs'][self.i_away]

    def away_turnover_xb(self):
        return self.t['atts'][self.i_away] + self.t['defs'][self.i_home] + self.t['home']

    def ex_t_offense_winning_greatly(self):
        return self.ex_t['offense_winning_greatly']

    def t_offense_winning_greatly(self):
        return self.t['offense_winning_greatly']

    def ex_t_offense_losing_badly(self):
        return self.ex_t['offense_losing_badly']

    def t_offense_losing_badly(self):
        return self.t['offense_losing_badly']

    def ex_t_two_minute_drill(self):
        return self.ex_t['two_minute_drill']

    def t_two_minute_drill(self):
        return self.t['two_minute_drill']

    def ex_t_baseline_hazards(self):
        return self.ex_t['baseline_hazards']

    def t_baseline_hazards(self):
        return self.t['baseline_hazards']


class ParamCalculatorNoTurnoverDefense(ParamCalculator):
    """
    No defense takeaway-propensity in the turnover model.
    """

    def __init__(self, ex_turnover, turnover):
        super(ParamCalculatorNoTurnoverDefense, self).__init__(ex_turnover, turnover)


    def re_draw_turnover_sample(self):
        num_samples = self.turnover.atts.gettrace().shape[0]
        draw = np.random.randint(0, num_samples)
        return {'atts': self.turnover.atts.gettrace()[draw, :],
                'home': self.turnover.home.gettrace()[draw],
                'baseline_hazards': self.turnover.baseline_hazards.gettrace()[draw, :],
                'two_minute_drill': self.turnover.two_minute_drill.gettrace()[draw],
                'offense_losing_badly': self.turnover.offense_losing_badly.gettrace()[draw],
                'offense_winning_greatly': self.turnover.offense_winning_greatly.gettrace()[draw]}

    def home_turnover_xb(self):
        return self.t['atts'][self.i_home]


    def away_turnover_xb(self):
        return self.t['atts'][self.i_away] + self.t['home']


class ParamCalculatorNoRZ(ParamCalculator):
    """
    Redzone is identical to other pieces in terms of team-specific attack/defense/params.
    """

    def __init__(self, ex_turnover, turnover):
        super(ParamCalculatorNoRZ, self).__init__(ex_turnover, turnover)

    def home_xb_rz(self):
        return self.home_xb()

    def away_xb_rz(self):
        return self.away_xb()

    def re_draw_ex_turnover_sample(self):
        num_samples = self.ex_turnover.atts.gettrace().shape[0]
        draw = np.random.randint(0, num_samples)
        return {'atts': self.ex_turnover.atts.gettrace()[draw, :],
                'defs': self.ex_turnover.defs.gettrace()[draw, :],
                'home': self.ex_turnover.home.gettrace()[draw, :],
                'baseline_hazards': self.ex_turnover.baseline_hazards.gettrace()[draw, :],
                'two_minute_drill': self.ex_turnover.two_minute_drill.gettrace()[draw],
                'offense_losing_badly': self.ex_turnover.offense_losing_badly.gettrace()[draw],
                'offense_winning_greatly': self.ex_turnover.offense_winning_greatly.gettrace()[draw]}


class ParamCalculatorNoRZNoTurnoverDefense(ParamCalculatorNoRZ, ParamCalculatorNoTurnoverDefense):
    def __init__(self, ex_turnover, turnover):
        super(ParamCalculatorNoRZNoTurnoverDefense, self).__init__(ex_turnover, turnover)


class ParamCalculatorNoRZNoTurnoverDefenseNeutralField(ParamCalculatorNoRZ, ParamCalculatorNoTurnoverDefense):
    def __init__(self, ex_turnover, turnover):
        super(ParamCalculatorNoRZNoTurnoverDefenseNeutralField, self).__init__(ex_turnover, turnover)

    def re_draw_ex_turnover_sample(self):
        num_samples = self.ex_turnover.atts.gettrace().shape[0]
        draw = np.random.randint(0, num_samples)
        return {'atts': self.ex_turnover.atts.gettrace()[draw, :],
                'defs': self.ex_turnover.defs.gettrace()[draw, :],
                'home': np.zeros(32),
                'baseline_hazards': self.ex_turnover.baseline_hazards.gettrace()[draw, :],
                'two_minute_drill': self.ex_turnover.two_minute_drill.gettrace()[draw],
                'offense_losing_badly': self.ex_turnover.offense_losing_badly.gettrace()[draw],
                'offense_winning_greatly': self.ex_turnover.offense_winning_greatly.gettrace()[draw]}

    def re_draw_turnover_sample(self):
        num_samples = self.turnover.atts.gettrace().shape[0]
        draw = np.random.randint(0, num_samples)
        return {'atts': self.turnover.atts.gettrace()[draw, :],
                'home': 0,
                'baseline_hazards': self.turnover.baseline_hazards.gettrace()[draw, :],
                'two_minute_drill': self.turnover.two_minute_drill.gettrace()[draw],
                'offense_losing_badly': self.turnover.offense_losing_badly.gettrace()[draw],
                'offense_winning_greatly': self.turnover.offense_winning_greatly.gettrace()[draw]}
        
        
def simulate_median_team_playing_schedule(season_df, teams, ex_turnover, turnover, param_calculator, n_per):
    """

    :param season_df:
    :param teams:
    :param ex_turnover:
    :param turnover:
    :param n_per:
    :return:
    """
    results = []
    teams_with_median = teams.copy()
    teams_with_median.loc[32, 'slug'] = 'MED'
    teams_with_median.loc[32, 'i'] = 32
    for i, row in teams.iterrows():
        print row['slug']
        df = season_df[(season_df.i_home == i) | (season_df.i_away == i)].copy()
        df.loc[df.i_home == i, 'i_home'] = MEDIAN_I
        df.loc[df.i_away == i, 'i_away'] = MEDIAN_I
        df2 = simulate_n_seasons(df, teams_with_median, ex_turnover, turnover, param_calculator, n_per)
        df2 = df2[df2.slug == 'MED']
        df2['schedule'] = row['slug']
        results.append(df2)
    return pd.concat(results, ignore_index=True)


def simulate_everyone_playing_median_team(teams, ex_turnover, turnover, param_calculator, n_per=100):
    """

    :param teams:
    :param ex_turnover:
    :param turnover:
    :param n_per:
    :return:
    """
    results = []
    for i, row in teams.iterrows():
        print row['slug']
        results.append(simulate_playing_median_team(i, teams, ex_turnover, turnover, param_calculator, n_per))
    return pd.concat(results, ignore_index=True)


def simulate_playing_median_team(i, teams, ex_turnover, turnover, param_calculator, n=100):
    """

    :param i:
    :param teams:
    :param ex_turnover:
    :param turnover:
    :param param_calculator:
    :param n:
    :return:
    """
    game_1 = {'hometeam': teams.slug[i],
              'awayteam': 'MED',
              'i_home': i,
              'i_away': MEDIAN_I}
    game_2 = {'hometeam': 'MED',
              'awayteam': teams.slug[i],
              'i_home': MEDIAN_I,
              'i_away': i}
    season_df = pd.DataFrame([game_1, game_2])
    return simulate_n_seasons(season_df, teams, ex_turnover, turnover, param_calculator, n=n)


def winner(x, home_advantage=SIMULATION_TIE_BREAKER_COIN_FLIP_HOME_ADVANTAGE):
    if x['home_score'] > x['away_score']:
        return x['hometeam']
    elif x['home_score'] < x['away_score']:
        return x['awayteam']
    return x['hometeam'] if x['coinflip'] > .5 - home_advantage else x['awayteam']


def simulate_n_seasons(season_df, teams, ex_turnover, turnover, param_calculator, n=100, collect_drives=False,
                       verbose=False):
    season_tables = []
    drives = []
    for i in range(n):
        if verbose and i and not i % 50:
            print '%s seasons simulated.' % i
        season_i, drive_stats = simulate_season(season_df, ex_turnover, turnover, param_calculator)
        # coinflip for ties
        season_i['coinflip'] = np.random.random(size=season_i.shape[0])
        season_i['winner'] = season_i.apply(winner, axis=1)
        season_i['home_win'] = (season_i.winner == season_i.hometeam).astype(int)
        season_i['home_loss'] = (season_i.winner == season_i.awayteam).astype(int)
        season_i['away_win'] = (season_i.winner == season_i.awayteam).astype(int)
        season_i['away_loss'] = (season_i.winner == season_i.hometeam).astype(int)
        season_table = create_season_table(season_i, teams)
        season_table['iteration'] = i
        season_tables.append(season_table)

        if collect_drives:
            drives += drive_stats

    df = pd.concat(season_tables, ignore_index=True)

    if collect_drives:
        df_drive = pd.DataFrame(drives)
        return df, df_drive
    else:
        return df


def create_season_table(season, teams):
    """
    Using a season dataframe output by simulate_season(), create a summary dataframe with wins, losses, goals for, etc.

    """
    g = season.groupby('i_home')
    home = pd.DataFrame({'home_yards': g.home_yards.sum(),
                         'home_yards_allowed': g.away_yards.sum(),
                         'home_wins': g.home_win.sum(),
                         'home_losses': g.home_loss.sum(),
                         'home_turnovers': g.home_turnovers.sum(),
                         'home_takeaways': g.away_turnovers.sum(),
                         'home_points': g.home_score.sum(),
                         'home_possessions': g.home_possessions.sum()
    })
    g = season.groupby('i_away')
    away = pd.DataFrame({'away_yards': g.away_yards.sum(),
                         'away_yards_allowed': g.home_yards.sum(),
                         'away_wins': g.away_win.sum(),
                         'away_losses': g.away_loss.sum(),
                         'away_turnovers': g.away_turnovers.sum(),
                         'away_takeaways': g.home_turnovers.sum(),
                         'away_points': g.away_score.sum(),
                         'away_possessions': g.away_possessions.sum()
    })
    df = home.join(away)
    df['wins'] = df.home_wins + df.away_wins
    df['losses'] = df.home_losses + df.away_losses
    df['yards'] = df.home_yards + df.away_yards
    df['yards_allowed'] = df.home_yards_allowed + df.away_yards_allowed
    df['turnovers'] = df.home_turnovers + df.away_turnovers
    df['takeaways'] = df.home_takeaways + df.away_takeaways
    df['points'] = df.home_points + df.away_points
    df['possessions'] = df.home_possessions + df.away_possessions
    df = pd.merge(teams, df, left_on='i', right_index=True, how='outer')
    return df


def simulate_season(season_df, ex_turnover, turnover, param_calculator):
    """
    Simulate a season once, using one random draw from the mcmc chain.
    """
    pc = param_calculator(ex_turnover, turnover)

    # for data collection
    season_simul = season_df.copy()
    stat_cols = []
    for pos in ['home', 'away']:
        for stat in ['score', 'yards', 'turnovers', 'possessions']:
            stat_cols.append('%s_%s' % (pos, stat))
    for col in stat_cols:
        season_simul[col] = None

    # simulate each game
    drives = []
    for i, row in season_simul.iterrows():
        i_home, i_away = row['i_home'], row['i_away']
        pc.i_home = i_home
        pc.i_away = i_away
        pc.re_draw_sample()
        results, drive_stats = simulate_game(pc)
        for c in stat_cols:
            season_simul.loc[i, c] = results[c]
        drives += drive_stats
    return season_simul, drives


def current_piece(yardline):
    for i in range(len(PIECES)):
        if PIECES[0] <= yardline < PIECES[i + 1]:
            return i
    raise Exception


def piece_lengths(PIECES):
    return [PIECES[i + 1] - PIECES[i] for i in range(len(PIECES) - 1)]


def simulate_game_n_times(pc, n=10000):
    results = []
    for i in range(n):
        pc.re_draw_sample()
        game_results, drive_stats = simulate_game(pc)
        game_results['i_home'] = pc.i_home
        game_results['i_away'] = pc.i_away
        results.append(game_results)
    return pd.DataFrame(results)


def simulate_game(params, verbose=False):
    """
    Simulate a set of drives using params drawn from posterior distributions.
    This is big and needs to be broken up.  #todo

    :param ex_turnover_params: dictionary of params drawn from ex_turnover posterior
    :param turnover_params: dictionary of params drawn from turnover posterior
    :param i_home: int team index of home team
    :param i_away: int team index of away team
    :param verbose: bool
    :return: game_stats dict and drive_stats list of dicts
    """
    i_home = params.i_home
    i_away = params.i_away


    # data collection
    score = {'home': 0, 'away': 0}
    offensive_yards = {'home': 0, 'away': 0}
    turnovers = {'home': 0, 'away': 0}
    possessions = {'home': 0, 'away': 0}
    drives = []

    clock = 60
    yardline = 20

    possession = 'home' if np.random.random() > .5 else 'away'
    defending = 'home' if possession == 'away' else 'away'

    first_half_possession = possession
    hit_halftime = False
    while clock > 0:  # BEGIN NEW DRIVE

        # new drive
        drive_in_progress = True
        possessions[possession] += 1
        drive_start = yardline
        total_drive_yards = 0
        outcome = None
        this_drive = {'start_yardline': yardline,
                      'i_home': i_home,
                      'i_away': i_away,
                      'i_attacking': i_home if possession == 'home' else i_away,
                      'i_defending': i_away if possession == 'home' else i_home,
                      'score_home': score['home'],
                      'score_away': score['away'],
                      'score_attacking': score['home'] if possession == 'home' else score['away'],
                      'score_defending': score['away'] if possession == 'home' else score['home'],
                      'start_clocktime': clock,
                      'two_minute_drill': 30 < clock < 32 or clock < 2
        }
        this_drive['offense_winning_greatly'] = (this_drive['score_attacking'] - this_drive[
            'score_defending']) > LOSING_BADLY_THRESHOLD
        this_drive['offense_losing_badly'] = (this_drive['score_defending'] - this_drive[
            'score_attacking']) > LOSING_BADLY_THRESHOLD

        if verbose:
            print '%s has the ball at the %s yardline with %s remaining.' % (possession, yardline, clock)

        while drive_in_progress:  # BEGIN NEW INTERVAL

            piece = current_piece(yardline)
            piece_start = PIECES[piece]
            piece_end = PIECES[piece + 1]

            # base xb for team and piece
            if piece == REDZONE_PIECE:
                xb = params.home_xb_rz() if possession == 'home' else params.away_xb_rz()
            else:
                xb = params.home_xb() if possession == 'home' else params.away_xb()
            xb_turn = params.home_turnover_xb() if possession == 'home' else params.away_turnover_xb()

            # amend xb for drive-specific situations
            if score[possession] - score[defending] > LOSING_BADLY_THRESHOLD:
                xb += params.ex_t_offense_winning_greatly()
                xb_turn += params.t_offense_winning_greatly()
            if score[defending] - score[possession] > LOSING_BADLY_THRESHOLD:
                xb += params.ex_t_offense_losing_badly()
                xb_turn += params.t_offense_losing_badly()
            if 30 < clock < 32 or clock < 2:
                xb += params.ex_t_two_minute_drill()
                xb_turn += params.t_two_minute_drill()

            # piece-specific baseline, and exp
            hazard = (params.ex_t_baseline_hazards()[piece]) * np.exp(xb)
            hazard_turnover = (params.t_baseline_hazards()[piece]) * np.exp(xb_turn)
            total_hazard = hazard + hazard_turnover

            # did they survive this piece?
            yards_survived = np.random.exponential(1. / total_hazard)


            # print '%s yards' % yards_survived

            if (yards_survived + yardline) > piece_end:  # survival to the next piece
                total_drive_yards += (piece_end - yardline)
                if piece == REDZONE_PIECE:
                    drive_in_progress = False
                    score[possession] += 7
                    clock -= TOUCHDOWN_CLOCK_TIME
                    yardline = 20
                    outcome = 'touchdown'
                    if verbose:
                        print '%s scored a touchdown.' % possession
                else:  # on to the next piece
                    if verbose:
                        print '%s advanced to the next piece by surviving %s yards' % (possession, yards_survived)
                    yardline = piece_end + .01
            else:  # drive death
                drive_in_progress = False
                total_drive_yards += yards_survived
                # was it 'normal' drive death, or death-by-turnover?
                p_turnover = hazard_turnover / (hazard_turnover + hazard)
                if np.random.random() < p_turnover:  # turnover
                    turnovers[possession] += 1
                    outcome = 'turnover'
                    if verbose:
                        print 'Turnover: %s gave away the ball at the %s yardline.' % (
                            possession, yards_survived + yardline)
                    yardline = 100 - (yards_survived + yardline)
                else:
                    # Punt or field goal?
                    if yards_survived + yardline > FIELD_GOAL_RANGE:  # field goal
                        clock -= FG_CLOCK_TIME
                        score[possession] += 3
                        outcome = 'field_goal'
                        if verbose:
                            print '%s kicked a field goal from the %s yardline' % (
                                possession, yardline + yards_survived)
                        yardline = 20
                    else:  # punt
                        clock -= PUNT_CLOCK_TIME
                        outcome = 'punt'
                        punt_to_yardline = 100 - (yardline + yards_survived + PUNT_DISTANCE)
                        punt_to_yardline = 20 if punt_to_yardline < 2 else punt_to_yardline  # cut down on coffin corner punts
                        if verbose:
                            print '%s punted from the %s yardline to the %s yardline.' % (
                                possession, yardline + yards_survived, punt_to_yardline)
                        yardline = punt_to_yardline

        if verbose:
            print '  -- total drive yards: %s' % total_drive_yards
        clock -= drive_time_elapsed(total_drive_yards)
        offensive_yards[possession] += total_drive_yards
        this_drive['end_clocktime'] = clock
        this_drive['end_yardline'] = drive_start + total_drive_yards
        this_drive['outcome'] = outcome
        this_drive['end_field_goal_attempt'] = outcome == 'field_goal'
        this_drive['end_touchdown'] = outcome == 'touchdown'
        drives.append(this_drive)

        # ALWAYS flip posession
        possession = 'away' if possession == 'home' else 'home'

        # handle the clock
        if clock < 30.5 and not hit_halftime:
            hit_halftime = True
            clock = 30
            possession = 'away' if first_half_possession == 'home' else 'home'
            yardline = 20

    game_stats = {'home_score': score['home'], 'away_score': score['away'],
                  'home_yards': offensive_yards['home'], 'away_yards': offensive_yards['away'],
                  'home_turnovers': turnovers['home'], 'away_turnovers': turnovers['away'],
                  'home_possessions': possessions['home'], 'away_possessions': possessions['away']}

    return game_stats, drives

##################################

import os
import sys


CHART_DIR = os.path.join(os.getcwd(), 'charts/')
sys.path.append(os.getcwd())

import math
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pymc as pm

CODE_DIR = os.path.join(os.getcwd(), 'code/')
DATA_DIR = os.path.join(os.getcwd(), 'data/')
DPI = 300
WIDECHARTWIDTH = 10
WIDECHARTHEIGHT = 6
SAVECHARTS = False


from nfl_hierarchical_bayes import data_prep, simulation_pwexp, elapsed_time, REDZONE_PIECE



# three tiers for attack/defense parameters
# no redzone breakout

df = pd.read_csv(DATA_DIR + 'pbp-2014-bugfixed.csv')
df = data_prep.enrich_play_level_df(df)
df = data_prep.merge_in_game_level_dataset(df, data_prep.GAME_LEVEL_DATASET_2014)
df = data_prep.calculate_game_score_at_play_start(df)
df_drive = data_prep.generate_drive_df(df)
df_drive, teams = data_prep.index_with_team_indexes(df_drive)
df_drive = data_prep.remove_unexplained_drives(df_drive)
df_drive = data_prep.enrich_drive_level_df(df_drive)

print 'Dropping %s drives due to their beginning with <30 seconds left in half' % \
      df_drive[(df_drive.thirty_seconds_drill)].shape[0]
df_drive = df_drive[~(df_drive.thirty_seconds_drill)]

print 'Dropping %s drives due to the ending with qb kneel' % df_drive[(df_drive.end_qb_kneel)].shape[0]
df_drive = df_drive[~(df_drive.end_qb_kneel)]

df_pw = data_prep.generate_piecewise_df(df_drive)
df_counts = data_prep.generate_piecewise_counts_df(df_pw)

observed_drive_deaths_ex_turnover = df_counts.deaths_ex_turnover.values
observed_exposures = df_counts.exposure_yards.values
piece_i = df_counts.piece_i.values
red_zone = (df_counts.piece_i == REDZONE_PIECE).astype(int).values
not_red_zone = (df_counts.piece_i != REDZONE_PIECE).astype(int).values
attacking_team = df_counts.i_attacking.values
defending_team = df_counts.i_defending.values
defending_team_is_home = df_counts.defending_team_is_home.values
offense_is_losing_badly = df_counts.offense_losing_badly.astype(int).values
offense_is_winning_greatly = df_counts.offense_winning_greatly.astype(int).values
drive_is_two_minute_drill = df_counts.two_minute_drill.astype(int).values
num_teams = len(df_counts.i_home.unique())
num_obs = len(drive_is_two_minute_drill)
num_pieces = len(df_counts.piece_i.unique())

observed_drive_deaths_turnover = df_counts.deaths_turnover.values

g = df_counts.groupby('piece_i')
baseline_starting_vals = g.deaths_ex_turnover.sum() / g.exposure_yards.sum()



def ex_turnover_piecewise_exponential_model():

    NCT_DOF = 4

    # hyperpriors for team-level distributions
    std_dev_att1 = pm.Uniform('std_dev_att1', lower=0, upper=50)
    std_dev_def1 = pm.Uniform('std_dev_def1', lower=0, upper=50)
    std_dev_att2 = pm.Uniform('std_dev_att2', lower=0, upper=50)
    std_dev_def2 = pm.Uniform('std_dev_def2', lower=0, upper=50)
    std_dev_att3 = pm.Uniform('std_dev_att3', lower=0, upper=50)
    std_dev_def3 = pm.Uniform('std_dev_def3', lower=0, upper=50)

    mu_att1 = pm.TruncatedNormal('mu_att1', 0, .001, -3, 0, value=-.2)
    mu_def1 = pm.TruncatedNormal('mu_def1', 0, .001, 0, 3, value=.2)
    mu_att3 = pm.TruncatedNormal('mu_att3', 0, .001, 0, 3, value=.2)
    mu_def3 = pm.TruncatedNormal('mu_def3', 0, .001, -3, 0, value=-.2)

    pi_att = pm.Dirichlet("grp_att", theta=[1,1,1])
    pi_def = pm.Dirichlet("grp_def", theta=[1,1,1])

    #team-specific parameters
    group_att = pm.Categorical('group_att', pi_att, size=num_teams)
    group_def = pm.Categorical('group_def', pi_def, size=num_teams)

    @pm.deterministic
    def mu_atts(group_att=group_att,
                mu_att1=mu_att1,
                mu_att3=mu_att3):
        mus_by_group = np.array([mu_att1, 0, mu_att3])
        return mus_by_group[group_att]

    @pm.deterministic
    def mu_defs(group_def=group_def,
                mu_def1=mu_def1,
                mu_def3=mu_def3):
        mus_by_group = np.array([mu_def1, 0, mu_def3])
        return mus_by_group[group_def]

    @pm.deterministic
    def tau_atts(group_att=group_att,
                std_dev_att1=std_dev_att1,
                std_dev_att2=std_dev_att2,
                std_dev_att3=std_dev_att3):
        taus_by_group = np.array([std_dev_att1**-2, std_dev_att2**-2, std_dev_att3**-2])
        return taus_by_group[group_att]


    @pm.deterministic
    def tau_defs(group_def=group_def,
                std_dev_def1=std_dev_def1,
                std_dev_def2=std_dev_def2,
                std_dev_def3=std_dev_def3):
        taus_by_group = np.array([std_dev_def1**-2, std_dev_def2**-2, std_dev_def3**-2])
        return taus_by_group[group_def]

    atts_star = np.empty(num_teams, dtype=object)
    defs_star = np.empty(num_teams, dtype=object)

    for i in range(num_teams):
        atts_star[i] = pm.NoncentralT("att_%i" % i, mu=mu_atts[i], lam=tau_atts[i], nu=NCT_DOF)
        defs_star[i] = pm.NoncentralT("def_%i" % i, mu=mu_defs[i], lam=tau_defs[i], nu=NCT_DOF)

    # home
    mu_home = pm.Normal('mu_home', 0, .0001, value=-.01)
    std_dev_home = pm.Uniform('std_dev_home', lower=0, upper=50)

    @pm.deterministic(plot=False)
    def tau_home(std_dev_home=std_dev_home):
        return std_dev_home**-2

    home = pm.Normal('home',
                     mu=mu_home,
                     tau=tau_home, size=num_teams, value=np.zeros(num_teams))

    # priors on coefficients
    baseline_hazards = pm.Normal('baseline_hazards', 0, .0001, size=num_pieces, value=baseline_starting_vals.values)
    two_minute_drill = pm.Normal('two_minute_drill', 0, .0001, value=-.01)
    offense_losing_badly = pm.Normal('offense_losing_badly', 0, .0001, value=-.01)
    offense_winning_greatly = pm.Normal('offense_winning_greatly', 0, .0001, value=.01)


    # trick to code the sum to zero contraint
    @pm.deterministic
    def atts(atts_star=atts_star):
        atts = [float(i) for i in atts_star]
        atts = atts - np.mean(atts)
        return atts

    @pm.deterministic
    def defs(defs_star=defs_star):
        defs = [float(i) for i in defs_star]
        defs = defs - np.mean(defs_star)
        return defs


    @pm.deterministic
    def mu_ijk(attacking_team=attacking_team,
                   defending_team=defending_team,
                   defending_team_is_home=defending_team_is_home,
                   two_minute_drill=two_minute_drill,
                   drive_is_two_minute_drill=drive_is_two_minute_drill,
                   offense_losing_badly=offense_losing_badly,
                   offense_is_losing_badly=offense_is_losing_badly,
                   offense_winning_greatly=offense_winning_greatly,
                   offense_is_winning_greatly=offense_is_winning_greatly,
                   home=home,
                   atts=atts,
                   defs=defs,
                   baseline_hazards=baseline_hazards,
                   observed_exposures=observed_exposures,
                   piece_i=piece_i):
        return  observed_exposures * baseline_hazards[piece_i] * \
                    np.exp(   home[defending_team] * defending_team_is_home + \
                              two_minute_drill * drive_is_two_minute_drill + \
                              offense_losing_badly * offense_is_losing_badly + \
                              offense_winning_greatly * offense_is_winning_greatly + \
                              atts[attacking_team] + \
                              defs[defending_team])


    drive_deaths = pm.Poisson("drive_deaths", mu_ijk,
                              value=observed_drive_deaths_ex_turnover, observed=True)

    @pm.potential
    def limit_sd(std_dev_att1=std_dev_att1,
                 std_dev_att2=std_dev_att2,
                 std_dev_att3=std_dev_att3,
                 std_dev_def1=std_dev_def1,
                 std_dev_def2=std_dev_def2,
                 std_dev_def3=std_dev_def3,
                 std_dev_home=std_dev_home):
        if std_dev_att1 < 0 or std_dev_att2 < 0 or std_dev_att3 < 0:
            return -np.inf
        if std_dev_def1 < 0 or std_dev_def2 < 0 or std_dev_def3 < 0:
            return -np.inf
        if std_dev_home < 0:
            return -np.inf
        return 0

    @pm.potential
    def keep_mu_within_bounds(mu_att1=mu_att1,
                              mu_def1=mu_def1,
                              mu_att3=mu_att3,
                              mu_def3=mu_def3):
        if mu_att1 < -3 or mu_att1 > 0 or mu_def3 < -3 or mu_def3 > 0:
            return -np.inf
        if mu_def1 < 0 or mu_def1 > 3 or mu_att3 < 0 or mu_att3 > 3:
            return -np.inf
        return 0

    return locals()


ex_turnover = pm.MCMC(ex_turnover_piecewise_exponential_model(),
                      db='pickle', dbname=DATA_DIR + 'ex_turnover_three_tiers_norz.pickle')