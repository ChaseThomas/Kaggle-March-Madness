import numpy as np
import pandas as pd
from pathlib import Path
import re
import os

AVG_FILEPATH = "ryan/data/processed/team_averages.csv"
MASSEY_FILEPATH = "ryan/data/processed/massey_ordinals.csv"
MASSEY_2017_FILEPATH = "ryan/data/processed/massey_ordinals_2017.csv"


def shooting(fgm, fga, fgm3):
    if fga == 0:
        return 0
    else:
        return (fgm + 0.5 * fgm3) / fga


def turnovers(to, fga, fta):
    tmp = fga + 0.44 * fta + to
    if tmp == 0:
        return 0
    else:
        return to / tmp


def off_rebounds(my_or, opp_dr):
    tmp = my_or + opp_dr
    if tmp == 0:
        return 0
    else:
        return my_or / tmp


def def_rebounds(my_dr, opp_or):
    tmp = my_dr + opp_or
    if tmp == 0:
        return 0
    else:
        return my_dr / tmp


def freethrows(ftm, fta):
    if fta == 0:
        return 0
    else:
        return ftm / fta


def process_row(x):
    w_shooting = shooting(x['Wfgm'], x['Wfga'], x['Wfgm3'])
    l_shooting = shooting(x['Lfgm'], x['Lfga'], x['Lfgm3'])
    w_turnovers = turnovers(x['Wto'], x['Wfga'], x['Wfta'])
    l_turnovers = turnovers(x['Lto'], x['Lfga'], x['Lfta'])
    w_off_rebounds = off_rebounds(x['Wor'], x['Ldr'])
    l_off_rebounds = off_rebounds(x['Lor'], x['Wdr'])
    w_def_rebounds = def_rebounds(x['Wdr'], x['Lor'])
    l_def_rebounds = def_rebounds(x['Ldr'], x['Wor'])
    w_freethrows = freethrows(x['Wftm'], x['Wfta'])
    l_freethrows = freethrows(x['Lftm'], x['Lfta'])
    result = {
        'Season': x['Season'],
        'Wteam': x['Wteam'],
        'Wdef_rebounds': w_def_rebounds,
        'Wfreethrows': w_freethrows,
        'Woff_rebounds': w_off_rebounds,
        'Wshooting': w_shooting,
        'Wturnovers': w_turnovers,
        'Lteam': x['Lteam'],
        'Ldef_rebounds': l_def_rebounds,
        'Lfreethrows': l_freethrows,
        'Loff_rebounds': l_off_rebounds,
        'Lshooting': l_shooting,
        'Lturnovers': l_turnovers,
    }
    return pd.Series(result, dtype=np.float32)


def load_tourney_results():
    print("Loading Tournament Results")
    filename = "ryan/data/TourneyCompactResults.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_full = pd.read_csv(
        filename,
        usecols=('Season', 'Wteam', 'Lteam'),
        dtype=np.float32,
        skipinitialspace=True,
        index_col='Season'
    )
    return df_full.reset_index()


def load_toruney_seeds():
    print("Loading Tournament Seeds")
    filename = "ryan/data/TourneySeeds.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_full = pd.read_csv(
        filename,
        skipinitialspace=True,
        index_col=['Season', 'Team']
    )
    df_full["Seed"] = df_full["Seed"].apply(lambda x: int(re.findall("\d+", x)[0]))
    return df_full.reset_index()


def preprocess_massey_2017():
    ordinals_previous_df = pd.read_csv("ryan/data/massey_ordinals_2017.csv", index_col=['season', 'team'],
                                       skipinitialspace=True)
    ordinals_previous_df = ordinals_previous_df[ordinals_previous_df.rating_day_num == 133]
    del ordinals_previous_df['rating_day_num']
    ordinals_previous_df = ordinals_previous_df.reset_index().groupby(['season', 'team'], as_index=True).mean()

    os.makedirs(os.path.dirname(MASSEY_2017_FILEPATH), exist_ok=True)
    ordinals_previous_df.to_csv(MASSEY_2017_FILEPATH)
    return ordinals_previous_df


def preprocess_massey():
    processed_file = Path(MASSEY_FILEPATH)
    if processed_file.is_file():
        print("Loading Massey Ordinals...")
        ordinals_df = pd.read_csv(processed_file, index_col=['season', 'team'], dtype=np.float32, skipinitialspace=True)
    else:
        print("Preprocessing Massey Ordinals...")
        ordinals_df = pd.read_csv("ryan/data/massey_ordinals_2003-2016.csv", index_col=['season', 'team'], skipinitialspace=True)
        ordinals_df = ordinals_df[ordinals_df.rating_day_num == 133]
        del ordinals_df['rating_day_num']
        ordinals_df = ordinals_df.reset_index().groupby(['season', 'team'], as_index=True).mean()
        ordinals_2017_df = preprocess_massey_2017()
        ordinals_df = pd.concat([ordinals_df, ordinals_2017_df])
        '''ordinals_previous_df = ordinals_previous_df[ordinals_previous_df.sys_name == 'BWE']
        del ordinals_previous_df['sys_name']
        ordinals_previous_df = ordinals_previous_df[ordinals_previous_df.rating_day_num == 133]
        del ordinals_previous_df['rating_day_num']'''
        '''ordinals_previous_df = ordinals_previous_df.reset_index()
        ordinals_previous_df = ordinals_previous_df[ordinals_previous_df['season'] >= 2010]
        ordinals_previous_df = ordinals_previous_df.set_index(['season', 'team'])'''

        os.makedirs(os.path.dirname(MASSEY_FILEPATH), exist_ok=True)
        ordinals_df.to_csv(MASSEY_FILEPATH)
        print("Finished preprocessing Massey Ordinals!")
    return ordinals_df.reset_index()


def preprocess_team_avg():
    processed_file = Path(AVG_FILEPATH)
    if processed_file.is_file():
        print("Loading team averages...")
        team_avgs_df = pd.read_csv(processed_file, index_col=[0, 1], dtype=np.float32, skipinitialspace=True)
    else:
        print("Calculating team averages...")
        df_full = pd.read_csv(
            "ryan/data/RegularSeasonDetailedResults.csv",
            usecols=('Season', 'Wteam', 'Lteam', 'Wfgm',  'Lfgm',  'Wfga',  'Lfga',  'Wfgm3', 'Lfgm3', 'Wto', 'Lto',
                     'Wfta',  'Lfta',  'Wor',   'Lor',   'Wdr',   'Ldr',   'Wftm',  'Lftm'),
            dtype=np.float32,
            skipinitialspace=True,
        )
        processed_df = df_full.apply(process_row, axis=1)
        w_split_df = processed_df.filter(axis=1, regex="^W")
        l_split_df = processed_df.filter(axis=1, regex="^L")
        w_split_df = pd.concat([processed_df['Season'], w_split_df], axis=1)
        l_split_df = pd.concat([processed_df['Season'], l_split_df], axis=1)

        wins = w_split_df.groupby(['Season', 'Wteam'], as_index=True).size().rename("Wins").rename_axis(['Season', 'Team'])
        losses = l_split_df.groupby(['Season', 'Lteam'], as_index=True).size().rename("Losses").rename_axis(['Season', 'Team'])
        w_avg_df = w_split_df.groupby(['Season', 'Wteam'], as_index=True).mean().rename_axis(['Season', 'Team'])
        l_avg_df = l_split_df.groupby(['Season', 'Lteam'], as_index=True).mean().rename_axis(['Season', 'Team'])

        team_avgs_df = pd.concat([wins, w_avg_df, losses, l_avg_df], axis=1)
        team_avgs_df = team_avgs_df.fillna(0.0)

        os.makedirs(os.path.dirname(AVG_FILEPATH), exist_ok=True)
        team_avgs_df.to_csv(AVG_FILEPATH)
        print("Finished calculating team averages!")

    return team_avgs_df.reset_index()


def dataframes_to_matricies(team_avgs_df, massey_ordinals_df, tourney_seeds_df, tourney_results_df):
    regular_matrix = team_avgs_df.reset_index().values
    tourney_matrix = tourney_results_df.reset_index().values
    tourney_matrix = tourney_matrix[tourney_matrix[:, 0] >= 2003.0]  # match season>=2003
    massey_matrix = massey_ordinals_df.reset_index().values
    seeds_matrix = tourney_seeds_df.reset_index().values

    # only incorporate data that has massey rankings for those seasons
    tourney_matrix = tourney_matrix[np.logical_or.reduce([tourney_matrix[:, 0] == x for x in np.asarray(massey_matrix[:, 0])])]
    regular_matrix = regular_matrix[np.logical_or.reduce([regular_matrix[:, 0] == x for x in np.asarray(massey_matrix[:, 0])])]
    seeds_matrix = seeds_matrix[np.logical_or.reduce([seeds_matrix[:, 0] == x for x in np.asarray(massey_matrix[:, 0])])]

    # initialize empty result matricies
    num_features = (regular_matrix.shape[1] - 2 + 1) + 1
    x_matrix = np.empty((tourney_matrix.shape[0] * 2, num_features), dtype=np.float32)
    y_matrix = np.empty((tourney_matrix.shape[0] * 2,))

    for row in range(tourney_matrix.shape[0]):
        # filter by season
        regular_matched_season = regular_matrix[regular_matrix[:, 0] == tourney_matrix[row, 0]]
        massey_matched_season = massey_matrix[massey_matrix[:, 0] == tourney_matrix[row, 0]]
        seeds_matched_season = seeds_matrix[seeds_matrix[:, 0] == tourney_matrix[row, 0]]

        # get avgs for the winning and losing teams
        w_team_avgs = regular_matched_season[regular_matched_season[:, 1] == tourney_matrix[row, 1]][0]
        l_team_avgs = regular_matched_season[regular_matched_season[:, 1] == tourney_matrix[row, 2]][0]

        # get massey ordinals for the winning and losing teams
        w_massey = massey_matched_season[massey_matched_season[:, 1] == tourney_matrix[row, 1]][0][2:]
        l_massey = massey_matched_season[massey_matched_season[:, 1] == tourney_matrix[row, 2]][0][2:]
        #delta_massey = w_massey - l_massey

        # get seeds for winning and losing teams
        w_seed = seeds_matched_season[seeds_matched_season[:, 1] == tourney_matrix[row, 1]][0][2:]
        l_seed = seeds_matched_season[seeds_matched_season[:, 1] == tourney_matrix[row, 2]][0][2:]

        # populate x_matrix with stats independent of victory state
        #x_matrix[2 * row] = w_massey-l_massey
        #x_matrix[2 * row + 1] = l_massey-w_massey

        x_matrix[2 * row] = np.concatenate((w_massey-l_massey, w_seed-l_seed, w_team_avgs[2:]-l_team_avgs[2:]))
        x_matrix[2 * row + 1] = np.concatenate((l_massey-w_massey, l_seed-w_seed, l_team_avgs[2:]-w_team_avgs[2:]))

        # assign victory status
        y_matrix[2 * row] = 1.0
        y_matrix[2 * row + 1] = 0.0

    return x_matrix, y_matrix
