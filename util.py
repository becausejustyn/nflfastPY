import codecs
import numpy as np
import pandas as pd
import config

class SeasonNotFoundError(Exception):
    pass

def convert_to_gsis_id(new_id):
    '''Convert new player id columns to old gsis id'''
    if type(new_id) == float:
        return new_id
        
    return codecs.decode(new_id[4:-8].replace('-',''),"hex").decode('utf-8')

def load_pbp_data(year=2021):
    '''
    Load NFL play by play data going back to 1999
    '''
    if type(year) is not int:
        raise TypeError('Please provide an integer between 1999 and 2021 for the year argument.')
    if year < 1999 or year > 2022:
        raise SeasonNotFoundError('Play by play data is only available from 1999 to 2022')
    df = pd.read_csv(config.pbp_loc.format(year=year), compression='gzip', low_memory=False)
    return df

#for seasons_ in range(season):
#        files = config.data_dir + f'pbp/csv/play_by_play_{seasons_}.csv.gz'
#        df = pd.concat((pd.read_csv(__, low_memory=False, index_col=0, compression='gzip') for __ in files))

def load_roster_data(year):
    '''Load team roster data 1999 -> 2021'''
    if type(year) is not int:
        raise TypeError('Please provide an integer between 1999 and 2021 for the year argument.')
    if year < 1999 or year > 2021:
        raise SeasonNotFoundError('Roster data is only available from 1999 to 2021')
    df = pd.read_csv(config.roster_loc.format(year=year), low_memory=False)
    return df