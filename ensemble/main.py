import combine_csvs
import math
import model

def main_cross_validated():
    combined = combine_csvs.combine_csvs({
        "train": ["season", "team1", "team2", "actual"]
    })
    data = combined["train"]

    split_size = int(math.ceil(len(data)*0.7))
    train = data[:split_size]
    test = data[split_size:]

    positives = model.run_ensemble(train, test)
    accurate_count = 0

    for i in xrange(0, len(positives)):
        if test["actual"].values[i] == positives[i]:
            accurate_count += 1
    print "Cross Validated Accuracy:"
    print accurate_count / len(test)

def main():
    combined = combine_csvs.combine_csvs({
        "train": ["season", "team1", "team2", "actual"],
        "test": ["team1", "team2"],
    })
    train = combined["train"]
    test = combined["test"]
    model.run_ensemble(train, test)

#main()

main_cross_validated()
