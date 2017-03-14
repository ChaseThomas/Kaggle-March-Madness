import pandas
from sklearn.linear_model import LogisticRegression

def run_ensemble(train, test):
    """
    Trains a logistic regression model on the "combined_train"
    and "combined_test" csv files in the "train" and "test" directories.
    """
    logR = LogisticRegression()
    logR = logR.fit(training.drop(['actual'], axis=1), training['actual'])
    logR.predict_proba(testing)
