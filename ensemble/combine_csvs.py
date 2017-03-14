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
    output = {}
    for directory in directories:
        columns = directories[directory]
        running_df = None
        filenames = [filename for filename in glob.glob("{}/*.csv".format(directory)) if "combined" not in filename]
        model_names = [filename.replace("{}/".format(directory), "").replace(".csv", "") for filename in filenames]
        for i in xrange(0, len(filenames)):
            filename = filenames[i]
            model_name = model_names[i]
            df = pandas.DataFrame.from_csv(filename, index_col=False)
            df = df[sorted(df.columns)]
            if i == 0:
                running_df = df
            else:
                running_df = running_df.join(
                    df.set_index(columns),
                    lsuffix="_{}".format(model_names[i-1]),
                    rsuffix="_{}".format(model_name),
                    on=columns
                )
        output[directory] = running_df
    return output
