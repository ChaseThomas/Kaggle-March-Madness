import csv
import glob
import pandas

def combine_csvs(directory):
    running_df = None
    for filename in glob.glob("{}/*.csv".format(directory)):
        df = pandas.DataFrame.from_csv(filename, index_col=False)
        df = df[sorted(df.columns)]
        if running_df is None:
            running_df = df
        else:
            running_df = running_df.append(df.copy())
    running_df.to_csv("{}/combined_{}.csv".format(directory, directory))

def combine_train_test():
    print "Combining training csvs..."
    combine_csvs("train")
    print "Combining testing csvs..."
    combine_csvs("test")

combine_train_test()
