import pandas
from sklearn.linear_model import LogisticRegression

def run_ensemble(train, test):
    """
    Trains a logistic regression model on the "combined_train"
    and "combined_test" csv files in the "train" and "test" directories.
    """
    logR = LogisticRegression()

    # Compose set of columns to use for training and testing

    columns = ['team1','team2']
    for col in train:
        if 'pred_' in col:
            columns.append(col)

    # Filter out NaN rows in training and testing set

    for col in columns:
        train = train[pandas.notnull(train[col])]
        test = test[pandas.notnull(test[col])]

    logR = logR.fit(train[columns], train['actual'])
    results = logR.predict_proba(test[columns])
    return results

def get_positives(results, threshold):
    """
    Uses the results from the 'run_ensemble' function above
    """
    positives = []
    # Each result corresponds to the probability of [-1, 1] in that order.
    # So need to check result[1] for each result in results
    for result in results:
        if result[1] > threshold:
            positives.append(1)
        else:
            positives.append(0)
    return positives
