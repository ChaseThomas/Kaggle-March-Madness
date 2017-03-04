import logistic_regression as lr
import evaluate as e
import csv
import preprocess as p
import decision_tree as d


def get_training_data_with_labels():
    with open('data/training_data.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile)
        train = []
        for row in reader:
            train.append(row)
    for i in range(len(train)):
        train[i] = map(float, train[i])
    return train

def get_training_data_wo_labels():
    with open('data/training_data.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile)
        train = []
        labels = []
        for row in reader:
            labels.append(row.pop(-1))
            train.append(row)
    for i in range(len(train)):
        train[i] = map(float, train[i])
    return train, labels


def main():
    # p.make_features()

    train_w_labels = get_training_data_with_labels()
    # train, labels = get_training_data_wo_labels()
    # Logistic Regression
    n_folds = 2
    # l_rate = 0.5
    # num_epochs = 100
    # print e.evaluate_algorithm(lr.logistic_regression, train_w_labels, n_folds, l_rate, num_epochs)

    # Decision Tree
    print e.evaluate_algorithm(d.decision_tree, train_w_labels, n_folds)




if __name__ == "__main__":
    main()