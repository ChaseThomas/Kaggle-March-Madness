from sklearn.model_selection import train_test_split
from ryan.preprocess import load_tourney_results
from ryan.preprocess import preprocess_team_avg
from ryan.preprocess import dataframes_to_matricies
from ryan.preprocess import load_toruney_seeds
from ryan.preprocess import preprocess_massey
from ryan.logistic_classifier import LogisticClassifier
from ryan.decision_tree import *
from sklearn import preprocessing


SEED = 31337


def main():
    massey_ordinals_df = preprocess_massey()
    tourney_results_df = load_tourney_results()
    tourney_seeds_df = load_toruney_seeds()
    team_avgs_df = preprocess_team_avg()
    #print(massey_ordinals_df.reset_index().groupby(['season']).size())
    #print(tourney_results_df.reset_index().groupby(['Season']).size())
    x_matrix, y_matrix = dataframes_to_matricies(team_avgs_df, massey_ordinals_df, tourney_seeds_df, tourney_results_df)
    # Split data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_matrix, test_size=0.3, random_state=SEED
    )

    print("Training size: %d, Testing size: %d" % (x_train.shape[0], x_test.shape[0]))

    print("Beginning Logistic Classifier Demo")
    classifier = LogisticClassifier(
        x_train, y_train,
        load_model="ryan/saved-networks/LogisticClassifier-2017-03-11_21-03-54.ckpt",
        num_epochs=0, beta=0.05, seed=SEED
    )
    print("Test Accuracy: %f" % classifier.test_accuracy(x_test, y_test))

    '''scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    classifier = LogisticRegression(scaler.transform(x_train), y_train, num_epochs=250000, beta=0.01, seed=SEED)
    print("Test Accuracy: %f" % classifier.test_accuracy(scaler.transform(x_test), y_test))'''

    classifier.save_model()

    '''print("Beginning Decision Tree Demo")
    tree = train_tree(x_train, y_train, seed=SEED)
    print("Finished training tree")

    print("Training Accuracy %f" % test_accuracy(tree, x_train, y_train))
    print("Testing Accuracy %f" % test_accuracy(tree, x_test, y_test))'''

if __name__ == "__main__":
    main()