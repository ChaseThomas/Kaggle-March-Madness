import numpy as np
import pandas as pd
from pathlib import Path

FILEPATH = "data/processed.csv"


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
        'def_rebounds': w_def_rebounds - l_def_rebounds,
        'freethrows': w_freethrows - l_freethrows,
        'off_rebounds': w_off_rebounds - l_off_rebounds,
        'shooting': w_shooting - l_shooting,
        'turnovers': w_turnovers - l_turnovers,
    }
    return pd.Series(result)


def preprocess():
    processed_file = Path(FILEPATH)
    if processed_file.is_file():
        print("Loading preprocessed data!")
        processed_df = pd.read_csv(processed_file, dtype=np.float32, skipinitialspace=True)
        print("Finished loading data!")
    else:
        print("Calculating data for preprocessing!")
        df_full = pd.read_csv(
            "data/RegularSeasonDetailedResults.csv",
            usecols=('Wfgm',  'Lfgm',  'Wfga',  'Lfga',  'Wfgm3', 'Lfgm3', 'Wto',   'Lto',
                     'Wfta',  'Lfta',  'Wor',   'Lor',   'Wdr',   'Ldr',   'Wftm',  'Lftm'),
            dtype=np.float32,
            skipinitialspace=True)
        processed_df = df_full.apply(process_row, axis=1)
        processed_df.to_csv(FILEPATH, index=False)
        print("Finished calculating data for preprocessing!")

    # Setup x_matrix
    x_matrix = processed_df.values
    x_matrix = np.repeat(x_matrix, 2, axis=0)  # Duplicate each row
    x_matrix[1::2, :] *= -1.0  # Invert every other row to represent losers

    # Setup y_matrix
    y_matrix = np.empty((x_matrix.shape[0], 1), dtype=np.float32)

    y_matrix[::2] = 1.0  # Winners
    y_matrix[1::2] = 0.0  # Losers

    return x_matrix, y_matrix
