import numpy as np
import pandas as pd
from pathlib import Path

AVG_FILEPATH = "data/team_averages.csv"


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
    processed_file = Path()
    print("Loading Tournament Results")
    df_full = pd.read_csv(
        "data/TourneyCompactResults.csv",
        usecols=('Wteam', 'Lteam'),
        dtype=np.float32,
        skipinitialspace=True
    )
    print(df_full)


def preprocess_team_avg():
    processed_file = Path(AVG_FILEPATH)
    if processed_file.is_file():
        print("Loading team averages!")
        team_avgs_df = pd.read_csv(processed_file, index_col=[0, 1], dtype=np.float32, skipinitialspace=True)
        print(team_avgs_df)
        print("Finished loading team averages!")
    else:
        print("Calculating team averages!")
        df_full = pd.read_csv(
            "data/RegularSeasonDetailedResults.csv",
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

        team_avgs_df.to_csv(AVG_FILEPATH)
        print("Finished calculating team averages!")

    return team_avgs_df
