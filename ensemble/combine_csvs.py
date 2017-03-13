import csv
import glob
import pandas

"""
Helper functions for combining CSV output files from individual models.
"""

def combine_csvs(directories):
    """
    Combines CSVs with the same column names in a given list of directories.
    Outputs one merged CSV per directory, with columns in each file
    sorted lexicographically by name.
    """
    for directory in directories:
        running_df = None
        for filename in glob.glob("{}/*.csv".format(directory)):
            df = pandas.DataFrame.from_csv(filename, index_col=False)
            df = df[sorted(df.columns)]
            if running_df is None:
                running_df = df
            else:
                running_df = running_df.append(df.copy())
        running_df.to_csv("{}/combined_{}.csv".format(directory, directory))

combine_csvs(["train", "test"])
