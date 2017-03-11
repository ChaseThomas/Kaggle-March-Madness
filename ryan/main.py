from sklearn.model_selection import train_test_split
from ryan.preprocess import load_tourney_results
from ryan.preprocess import preprocess_team_avg
from ryan.preprocess import dataframes_to_matricies
from ryan.preprocess import load_toruney_seeds
from ryan.preprocess import preprocess_massey
from ryan.logistic_regression import LogisticRegression
from ryan.decision_tree import *
from sklearn import preprocessing


SEED = 31337


def main():
    massey_ordinals_df = preprocess_massey()
    tourney_results_df = load_tourney_results()
    tourney_seeds_df = load_toruney_seeds()
    team_avgs_df = preprocess_team_avg()
    x_matrix, y_matrix = dataframes_to_matricies(team_avgs_df, massey_ordinals_df, tourney_seeds_df, tourney_results_df)
    # Split data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_matrix, test_size=0.3, random_state=SEED
    )

    print("Beginning Logistic Regression Demo")
    regression = LogisticRegression(x_train, y_train, num_epochs=500000, beta=0.01, seed=SEED)
    print("Test Accuracy: %f" % regression.test_accuracy(x_test, y_test))

    '''scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    regression = LogisticRegression(scaler.transform(x_train), y_train, num_epochs=250000, beta=0.01, seed=SEED)
    print("Test Accuracy: %f" % regression.test_accuracy(scaler.transform(x_test), y_test))'''

    regression.save_model()

    '''print("Beginning Decision Tree Demo")
    tree = train_tree(x_train, y_train, seed=SEED)
    print("Finished training tree")

    print("Training Accuracy %f" % test_accuracy(tree, x_train, y_train))
    print("Testing Accuracy %f" % test_accuracy(tree, x_test, y_test))'''

if __name__ == "__main__":
    main()