import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 1337

df_full = pd.read_csv("data/RegularSeasonDetailedResults.csv", skipinitialspace=True)

df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=SEED)

def shooting(fgm, fga, fgm3):
    return (fgm+0.5*fgm3) / fga

def turnovers(to, fga, fta):
    return to / (fga + 0.44 * fta + to)

def off_rebounds(my_or, opp_dr):
    return my_or / (my_or + opp_dr)

def def_rebounds(my_dr, opp_or):
    return my_dr / (my_dr + opp_or)

def freethrows(ftm, fta):
    return ftm / fta

def seed(seed):
    pass