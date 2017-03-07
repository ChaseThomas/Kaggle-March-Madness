from sklearn.model_selection import train_test_split
from ryan.preprocess import load_tourney_results
from ryan.preprocess import preprocess_team_avg
from ryan.logistic_regression import LogisticRegression
from ryan.decision_tree import *

SEED = 1337


def main():
    # load_tourney_results()
    preprocess_team_avg()
    # Split data into testing and training sets
''' x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_matrix, test_size=0.2, random_state=SEED
    )'''
'''
    print("Beginning Logistic Regression Demo")
    regression = LogisticRegression(x_train, y_train, num_epochs=100, seed=SEED)
    print("Test Accuracy: %f" % regression.test_accuracy(x_test, y_test))
    # regression.save_model()
'''

'''print("Beginning Decision Tree Demo")
    tree = train_tree(x_train, y_train, seed=SEED)
    print("Finished training tree")

    print("Training Accuracy %f" % test_accuracy(tree, x_train, y_train))
    print("Testing Accuracy %f" % test_accuracy(tree, x_test, y_test))'''

if __name__ == "__main__":
    main()