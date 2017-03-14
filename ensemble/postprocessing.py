import csv

def prepare_kaggle_csv(test, results):
    """
    Takes the test data columns and the result probabilities
    obtained from the run_ensemble() method.

    Converts these results to the format that Kaggle wants in its
    CSV files, with an 'id' and 'pred' column. Then it writes
    this data to a CSV file called 'submission.csv'.
    """
    csv_rows = [['id', 'pred']]

    for i in xrange(len(test)-1):
        kaggle_id = "2017_{}_{}".format(test["team1"].values[i], test["team2"].values[i])
        pred = results[i][0]
        csv_rows.append([kaggle_id, pred])

    with open('submission.csv', 'w') as kaggle_csv:
        writer = csv.writer(kaggle_csv)
        for row in csv_rows:
            writer.writerow(row)
