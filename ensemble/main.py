import math
import model
import preprocessing
import postprocessing

VALIDATION_SIZE = 0.3
MONTE_CARLO_ITERATIONS = 100

def main_cross_validated():
    """
    Uses Monte Carlo (repeating random subsampling) cross validation
    to access accuracy.

    Takes random splits of the dataset and uses them as training and
    validation sets. Records cross validation and averages recorded
    values over all iterations.
    """
    print "Preprocessing..."
    combined = preprocessing.combine_csvs({
        "train": ["season", "team1", "team2", "actual"]
    })
    data = combined["train"]

    cv_sum = 0

    print "Running cross validation..."
    for i in xrange(MONTE_CARLO_ITERATIONS):
        train = data.sample(frac = 1-VALIDATION_SIZE)
        test = data.sample(frac = VALIDATION_SIZE)

        positives = model.get_positives(
            model.run_ensemble(train, test),
            threshold=0.5
        )
        accurate_count = 0

        for i in xrange(0, len(positives)):
            if test["actual"].values[i] == positives[i]:
                accurate_count += 1
        cv = (accurate_count * 1.0) / len(test)
        cv_sum += cv

    cv_avg = cv_sum / MONTE_CARLO_ITERATIONS
    print "Monte Carlo cross validation accuracy:"
    print cv_avg

def main():
    """
    Runs the ensemble and saves the output in a ready-to-submit format.
    """
    print "Preprocessing..."
    combined = preprocessing.combine_csvs({
        "train": ["season", "team1", "team2", "actual"],
        "test": ["team1", "team2"],
    })
    train = combined["train"]
    test = combined["test"]
    print "Running logistic regression model..."
    probs = model.run_ensemble(train, test)
    print "Saving predictions..."
    postprocessing.prepare_kaggle_csv(test, probs)

# main()

main_cross_validated()
