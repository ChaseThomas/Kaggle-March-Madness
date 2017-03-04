from sklearn import tree


def train(features, labels):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    return clf


def predict(d_tree, test):
    return d_tree.predict(test)


def decision_tree(training_data, test):
    labels = []
    training_data_copy = []
    for row in training_data:
        this_row = list(row)
        labels.append(this_row.pop(-1))
        training_data_copy.append(this_row)

    dt = train(training_data_copy, labels)
    # remove labels from test data
    test_copy = []
    for row in test:
        this_row = list(row)
        this_row.pop(-1)
        test_copy.append(this_row)

    return predict(dt, test_copy)