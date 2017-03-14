import csv
import glob
import pandas

"""
Helper functions for combining CSV output files from individual models.
"""

def combine_csvs(directories):
    """
    Takes in a given dictionary of directory : column_list pairs.

    Each directory is expected to contain CSV files with the same set of columns.
    The CSVs are joined on the columns specified by column_list.

    Outputs a dictionary of directory : data_frame pairs
    where each data_frame contains the merged data.
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
            # Uncomment the line below to save results to CSV
            # running_df.to_csv("{}/combined_{}.csv".format(directory, directory))
        output[directory] = running_df
    return output
